import os
import json
from itertools import groupby

#from skimage.io import imread, imshow
from skimage import io
#from skimage.draw import disk
from skimage.morphology import (erosion, dilation, closing, opening, area_closing, area_opening)
#from skimage.color import rgb2gray

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from scipy.interpolate import BSpline, make_interp_spline
from scipy.cluster.hierarchy import fclusterdata
from scipy.signal import find_peaks, medfilt

from collections import defaultdict 
import numpy as np
import math
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')

class LidarPreprocessor:
    degrees1 = 268
    phi1 = (np.pi * degrees1) / 180.0
    SIN1 = np.sin(phi1)
    COS1 = np.cos(phi1)

    tx1 = 3833 + 85 #+ 108
    ty1 = 3497 + 35 #- 174
    HH1 = np.array([
        [COS1, -SIN1, tx1],
        [SIN1, COS1, ty1],
        [0, 0, 1]
    ])

    degrees2 = 180 # 183
    phi2 = (np.pi * degrees2) / 180.0
    SIN2 = np.sin(phi2)
    COS2 = np.cos(phi2)

    tx2 = 7730 # 7859 #+ 108
    ty2 = 3525-340 # 3226 #- 174
    HH2 = np.array([
        [COS2, -SIN2, tx2],
        [SIN2, COS2, ty2],
        [0, 0, 1]
    ])
    
    STRUCTURING_ELEMENT = np.array([
        [0,1,0],
        [1,1,1],
        [0,1,0]
    ])
    
    SE_RADIUS = 3

    
    @staticmethod
    def merge_lidar_dataframes(dfs):
        df1, df2, df3 = dfs

        # merge dataframes on timestamps
        df_merged = pd.merge_asof(df1, df2, on='timestamp')
        df_merged = pd.merge_asof(df_merged, df3, on='timestamp')    
        df_merged = df_merged.rename(columns={
            "fgx_x": "fgx_df1", 
            "fgy_x": "fgy_df1",
            "fgx_y": "fgx_df2", 
            "fgy_y": "fgy_df2",
            "fgx": "fgx_df3", 
            "fgy": "fgy_df3",
            "x_x": "x_df1",
            "y_x": "y_df1",
            "x_y": "x_df2",
            "y_y": "y_df2",
            "x": "x_df3",
            "y": "y_df3",
        })

        return df_merged

    # TODO: rename to calculate_affine_transformation
    @staticmethod
    def calculate_homogeneous_transformation(points_im1, points_im2):
        '''
        Computes Homogeneous transformation H. Points from im1 are transformed to the coordinate system of im2.
        To compute H, provide 3 point correspondences.
        @param points_im1 [collection<number>] 3 points
        @param points_im2 [collection<number>] 3 points
        @return 3x3 Homogeneous transformation
        '''
        x1, y1, x2, y2, x3, y3 = points_im1
        A = [
            [x1, y1, 1, 0, 0, 0],
            [0, 0, 0, x1, y1, 1],
            [x2, y2, 1, 0, 0, 0],
            [0, 0, 0, x2, y2, 1],
            [x3, y3, 1, 0, 0, 0],
            [0, 0, 0, x3, y3, 1]
        ]
        A = np.array(A, dtype=np.float32)
        v = np.array(points_im2)

        Asq = np.matmul(A.transpose(), A)
        Asq_inv = np.linalg.inv(Asq)
        A_pseud = np.matmul(Asq_inv, A.transpose())
        a, b, c, d, e, f = A_pseud.dot(v)

        H = [
            [a, b, c],
            [d, e, f],
            [0, 0, 1]
        ]

        H = np.array(H, dtype=np.float32)
        return H#np.linalg.inv(H)
    
    @staticmethod
    def calculate_rigid_transformation(points_im1, points_im2):
        '''
        Computes Homogeneous transformation H. Points from im1 are transformed to the coordinate system of im2.
        To compute H, provide 3 point correspondences.
        @param points_im1 [collection<number>] 3 points
        @param points_im2 [collection<number>] 3 points
        @return 3x3 Homogeneous transformation
        '''
        x1, y1, x2, y2 = points_im1
        A = [
            [x1, -y1, 1, 0],
            [y1, x1, 0, 1],
            [x2, -y2, 1, 0],
            [y2, x2, 0, 1],
        ]
        A = np.array(A, dtype=np.float32)
        v = np.array(points_im2)

        Asq = np.matmul(A.transpose(), A)
        Asq_inv = np.linalg.inv(Asq)
        A_pseud = np.matmul(Asq_inv, A.transpose())
        a, b, c, d = A_pseud.dot(v)

        H = [
            [a, -b, c],
            [b, a, d],
            [0, 0, 1]
        ]

        H = np.array(H, dtype=np.float32)
        return H#np.linalg.inv(H)
    
    @staticmethod
    def apply_homogeneous_transformation(xs, ys, H):
        '''
        @param xs coordnates
        @param ys coordinates
        @param H homogeneous transformation
        @return algined coordinates
        '''
        xy = np.array([xs, ys]).T
        xy = np.hstack((xy, np.ones((xy.shape[0], 1))))
        xy = np.dot(xy, H.T)
        xy = xy[:, 0:2] # remove the z-coordinates

        return xy[:,0], xy[:,1]
    
    @staticmethod
    def align_dataframes(df_merged, frame_id, use_foreground=True):
        column_name1 = "fgx_df1"
        column_name2 = "fgy_df1"
        column_name3 = "fgx_df2"
        column_name4 = "fgy_df2"

        if not use_foreground:
            column_name1 = "x_df1"
            column_name2 = "y_df1"
            column_name3 = "x_df2"
            column_name4 = "y_df2"

        H21 = LidarPreprocessor.calculate_homogeneous_transformation(ps11, ps21)
        H31 = LidarPreprocessor.calculate_homogeneous_transformation(ps1, ps2)

        aligned_xs1x, aligned_ys1x = LidarPreprocessor.apply_homogeneous_transformation(
            df_merged.iloc[frame_id][column_name1], df_merged.iloc[frame_id][column_name2], H21
        )

        aligned_xs2x, aligned_ys2x = LidarPreprocessor.apply_homogeneous_transformation(
            df_merged.iloc[frame_id][column_name3], df_merged.iloc[frame_id][column_name4], H31
        )
        
        return aligned_xs1x, aligned_ys1x, aligned_xs2x, aligned_ys2x
    
    @staticmethod
    def align_dataframes_with_homographies(df_merged, frame_id, H1, H2, use_foreground=True):
        column_name1 = "fgx_df1"
        column_name2 = "fgy_df1"
        column_name3 = "fgx_df2"
        column_name4 = "fgy_df2"

        if not use_foreground:
            column_name1 = "x_df1"
            column_name2 = "y_df1"
            column_name3 = "x_df2"
            column_name4 = "y_df2"

        aligned_xs1x, aligned_ys1x = LidarPreprocessor.apply_homogeneous_transformation(
            df_merged.iloc[frame_id][column_name1], df_merged.iloc[frame_id][column_name2], H1
        )

        aligned_xs2x, aligned_ys2x = LidarPreprocessor.apply_homogeneous_transformation(
            df_merged.iloc[frame_id][column_name3], df_merged.iloc[frame_id][column_name4], H2
        )
        
        return aligned_xs1x, aligned_ys1x, aligned_xs2x, aligned_ys2x 

    @staticmethod
    def multi_dil(im, num, element=STRUCTURING_ELEMENT):
        for i in range(num):
            im = dilation(im, element)
        return im
    
    @staticmethod
    def multi_ero(im, num, element=STRUCTURING_ELEMENT):
        for i in range(num):
            im = erosion(im, element)
        return im
    
    @staticmethod
    def multi_closing(im, num, element=STRUCTURING_ELEMENT):
        for i in range(num):
            im = closing(im, element)
        return im
    
    @staticmethod
    def multi_opening(im, num, element=STRUCTURING_ELEMENT):
        for i in range(num):
            im = opening(im, element)
        return im
    
    @staticmethod
    def create_frame_imgs_from_df(df, save_path, number_of_images=100):
        '''
        @param frames
        @param save_path "p64/L1904940"
        '''

        if not os.path.exists(save_path):
            os.mkdir(save_path)
            print(f"Directory {save_path} created")


        for fdx in range(number_of_images):
            row = df.iloc[fdx]
            x, y = row["x"], row["y"]
            fig = plt.figure(figsize=(6.4, 4.8))
            plt.axis('off')
            plt.scatter(x, y, alpha=1)
            plt.savefig(os.path.join(save_path, f"frame{fdx}.png"))
            plt.close(fig)

        return save_path
    
    @staticmethod
    def load_lidar_scan_images(base_path, img_count=100):
        '''
        @param fileapth "p64/L1904940/"
        '''
        return [io.imread(os.path.join(base_path, f"frame{idx}.png")) for idx in range(img_count)]

    @staticmethod
    def remove_background_from_frames_opt(df, mask_aaa, mask_bbb):
        x_coordinates, y_coordinates = df["x"].to_numpy(), df["y"].to_numpy()

        minimal_expected_distance_mm = 300
        def remove_background(x, y):
            _xss = np.zeros(len(x))
            _yss = np.zeros(len(y))

            # compute this using tensors
            for pix, p in enumerate(zip(x, y)):
                v = np.min(np.sqrt((mask_aaa - p[0])**2 + (mask_bbb - p[1])**2))

                if v > minimal_expected_distance_mm:
                    _xss[pix] = x[pix]
                    _yss[pix] = y[pix]

            return _xss, _yss


        clean_xs = []
        clean_ys = []
        for k in range(len(x_coordinates)):
            clean_x, clean_y = remove_background(x_coordinates[k], y_coordinates[k])
            print(f"Removed background for frame {k+1} / {len(x_coordinates)}")

            clean_xs.append(clean_x)
            clean_ys.append(clean_y)

        return clean_xs, clean_ys

    @staticmethod
    def compute_background_mask(df, imgs, min_density=0.4, max_density=0.6, img_count=100):
        erodeds = []
        for idx in range(img_count):
            img = imgs[idx]
            binary = 0.2125 * img[:,:,0] + 0.7154 * img[:,:,1] + 0.0721 * img[:,:,2]
            eroded = LidarPreprocessor.multi_closing(
                binary, LidarPreprocessor.SE_RADIUS, LidarPreprocessor.STRUCTURING_ELEMENT
            )
            erodeds.append(eroded)

        min_x = 10_000
        min_y = 10_000
        max_x = -1
        max_y = -1

        for fdx in range(img_count):
            row = df.iloc[fdx]
            x, y = row["x"], row["y"]
            if np.min(x) < min_x:
                min_x = np.min(x)

            if np.min(y) < min_y:
                min_y = np.min(y)

            if np.max(x) > max_x:
                max_x = np.max(x)

            if np.max(y) > max_y:
                max_y = np.max(y)


        mean_erodeds = np.mean(erodeds, axis=0)/255.0
        T = (1 - mean_erodeds)
        T = (T > min_density) & (T < max_density)

        # obtain nonzero coordinates of 270 clock-wise rotated mask.
        a, b = np.rot90(T, 3).nonzero()

        # transform index coordinates (a, b) to [0, 1]^2 space
        aa = a - np.min(a)
        aa = aa / np.max(aa)

        bb = b - np.min(b)
        bb = bb / np.max(bb)

        # transform [0, 1] index coordinates to [min_x, max_x] x [min_y, max_y] space
        aaa = aa * (max_x - min_x) + (min_x)
        bbb = bb * (max_y - min_y) + (min_y)

        mask = np.array(zip(aaa, bbb))

        return aaa, bbb, mask
    
    @staticmethod
    def compute_xy_for_df(df):
        frames = df["sample"].to_numpy()
        step_nb = len(frames[0])
        angle_min = math.radians((-270 / 2))
        angle_increment = math.radians(270) / step_nb
        angles = angle_min + np.arange(step_nb) * angle_increment

        xs = []
        ys = []

        max_scan_distance_mm = 7_000
        for frame in frames:
            filtered_frame = frame.copy()  
            filtered_frame[np.where(filtered_frame > max_scan_distance_mm)] = 0
            xy = filtered_frame * np.array([np.cos(angles), np.sin(angles)])
            xs.append(xy[0])
            ys.append(xy[1])
            
        return xs, ys
    
    @staticmethod
    def save_foreground_files(foreground_xs, foreground_ys, basepath, prefix):
        with open(os.path.join(basepath, f"{prefix}_ok_foreground_xs.npy"), 'wb') as file:
            np.save(file, foreground_xs)

        with open(os.path.join(basepath, f"{prefix}_ok_foreground_ys.npy"), 'wb') as file:
            np.save(file, foreground_ys)
    
    @staticmethod
    def preprocess_walk(dataset, save_df=False):
        parquet_filepath1, parquet_filepath2, parquet_filepath3 = dataset["filepaths"]
        participant = dataset["participant"]
        walk_uuid = dataset["walk-uuid"]
        trial_nr = dataset["trial-nr"]

        # load dataframes
        df1 = pd.read_parquet(parquet_filepath1, engine='pyarrow')
        df2 = pd.read_parquet(parquet_filepath2, engine='pyarrow')
        df3 = pd.read_parquet(parquet_filepath3, engine='pyarrow')
        df1 = df1.reset_index()
        df2 = df2.reset_index()
        df3 = df3.reset_index()

        # Compute cartestion coordinates from polar coordinates
        print("Computing Cartesian coordinates...")
        df1['x'], df1['y'] = LidarPreprocessor.compute_xy_for_df(df1)
        df2['x'], df2['y'] = LidarPreprocessor.compute_xy_for_df(df2)
        df3['x'], df3['y'] = LidarPreprocessor.compute_xy_for_df(df3)

        # Create images
        print("Creating frame iamges...")
        img_df1_path = LidarPreprocessor.create_frame_imgs_from_df(df1, save_path=os.path.join(participant, "200"))
        img_df2_path = LidarPreprocessor.create_frame_imgs_from_df(df2, save_path=os.path.join(participant, "201"))
        img_df3_path = LidarPreprocessor.create_frame_imgs_from_df(df3, save_path=os.path.join(participant, "203"))

        # Compute background masks
        print("Computing Background masks...")
        imgs_df1 = LidarPreprocessor.load_lidar_scan_images(base_path=img_df1_path)
        mask_aaa1, mask_bbb1, _ = LidarPreprocessor.compute_background_mask(df1, imgs_df1, min_density=0.41, max_density=0.45) # L3

        imgs_df2 = LidarPreprocessor.load_lidar_scan_images(base_path=img_df2_path)
        mask_aaa2, mask_bbb2, _ = LidarPreprocessor.compute_background_mask(df2, imgs_df2, min_density=0.41, max_density=0.45) # L3

        imgs_df3 = LidarPreprocessor.load_lidar_scan_images(base_path=img_df3_path)
        mask_aaa3, mask_bbb3, _ = LidarPreprocessor.compute_background_mask(df3, imgs_df3, min_density=0.41, max_density=0.45) # L3

        # Extract foreground from lidar scans
        print("Removing background from scans...")
        df1['fgx'], df1['fgy'] = LidarPreprocessor.remove_background_from_frames_opt(df1, mask_aaa1, mask_bbb1)
        df2['fgx'], df2['fgy'] = LidarPreprocessor.remove_background_from_frames_opt(df2, mask_aaa2, mask_bbb2)
        df3['fgx'], df3['fgy'] = LidarPreprocessor.remove_background_from_frames_opt(df3, mask_aaa3, mask_bbb3)

        out_parquet_filepath1 = os.path.join(participant, f"df_{participant}_{walk_uuid}_{trial_nr}_200.parquet")
        out_parquet_filepath2 = os.path.join(participant, f"df_{participant}_{walk_uuid}_{trial_nr}_201.parquet")
        out_parquet_filepath3 = os.path.join(participant, f"df_{participant}_{walk_uuid}_{trial_nr}_203.parquet")

        # Persist parquet files
        if save_df:
            print("Saving updates parquet files...")
            df1.to_parquet(out_parquet_filepath1)
            df2.to_parquet(out_parquet_filepath2)
            df3.to_parquet(out_parquet_filepath3)

        return df1, df2, df3
    
class LidarClustering:
    @staticmethod
    def cluster_point_cloud(X, max_tolerated_distance_in_mm):
        motion_groupings = []
        cluster_centers = []

        labels = fclusterdata(X, max_tolerated_distance_in_mm, criterion='distance', method="median")

        cluster_count = 0
        for k in set(labels):
            L_k = X[np.where(labels==k)[0]]
            xs_lk, ys_lk = list(zip(*L_k))
            if len(L_k) < 500 and len(L_k) > 10:
                cluster_count += 1
                cluster_centers.append([np.mean(xs_lk), np.mean(ys_lk)])
                motion_groupings.append([xs_lk, ys_lk])

        return motion_groupings, cluster_centers, cluster_count
    
    @staticmethod
    def group_motions_in_frame(df_merged, frame_id, apply_filtering=True):
        try:
            aligned_fgxs1x, aligned_fgys1x, aligned_fgxs2x, aligned_fgys2x = LidarPreprocessor.align_dataframes_with_homographies(
                df_merged, frame_id, LidarPreprocessor.HH1, LidarPreprocessor.HH2
            )
            fgx_df3, fgy_df3 = df_merged.iloc[frame_id]["fgx_df3"], df_merged.iloc[frame_id]["fgy_df3"]

            fg_xs = np.concatenate((aligned_fgxs1x, aligned_fgxs2x, fgx_df3))
            fg_ys = np.concatenate((aligned_fgys1x, aligned_fgys2x, fgy_df3))

            X = np.array(list(zip(fg_xs, fg_ys)))
            if apply_filtering:
                X = np.array(list(filter(
                    lambda x: (x[1] > 1500) and (x[1] < 3300) and (x[0] < 7710) and (x[0] > 400), X
                )))

            if len(X) == 0:
                return [], [], 0

            #TODO: refactor
            motion_groupings, cluster_centers, cluter_count = LidarClustering.cluster_point_cloud(
                X, max_tolerated_distance_in_mm=200
            )
            if cluter_count == 1:
                motion_groupings, cluster_centers, cluter_count = LidarClustering.cluster_point_cloud(
                    X, max_tolerated_distance_in_mm=175
                )
                if cluter_count == 1:
                    motion_groupings, cluster_centers, cluter_count = LidarClustering.cluster_point_cloud(
                        X, max_tolerated_distance_in_mm=150
                    )


            return motion_groupings, cluster_centers, cluter_count

        except ValueError as err:
            print(err)
            print(f"Error, no cluster for frame {frame_id}")
            return [], [], 0
    
    @staticmethod
    def split_cluster_into_two(grouped_points):
        # correction, if there is only one cc detected  

        xs_lk, ys_lk = grouped_points[0]
        X = np.array(list(zip(*(xs_lk, ys_lk))))

        k_means = KMeans(init='k-means++', n_clusters=2, n_init=1)
        k_means.fit(X)
        k_means_labels = k_means.labels_
        k_means_cluster_centers = k_means.cluster_centers_
        k_means_labels_unique = np.unique(k_means_labels)

        new_grouped_points = []
        new_cluster_centers = []

        labels = list(set(k_means_labels))

        x0 = X[np.where(k_means_labels==labels[0])]
        x01, x02 = list(zip(*x0))

        x1 = X[np.where(k_means_labels==labels[1])]
        x11, x12 = list(zip(*x1))

        new_grouped_points.append([x01, x02])
        new_grouped_points.append([x11, x12])

        new_cluster_centers.append([np.mean(x01), np.mean(x02)])
        new_cluster_centers.append([np.mean(x11), np.mean(x12)])

        return new_grouped_points, new_cluster_centers, 2
    
    @staticmethod
    def group_points(local_points, max_distance):
        groups = []

        points = local_points.copy()

        while points:
            far_points = []
            ref = points.pop()

            groups.append([ref])
            for point in points:
                d = LidarClustering.get_distance(ref, point)
                if d < max_distance:
                    groups[-1].append(point)
                else:
                    far_points.append(point)

            points = far_points

        # perform average operation on each group
        return groups, [list(np.mean(x, axis=1).astype(int)) for x in groups]
    
    @staticmethod
    def get_distance(ref, point):
        x1, y1 = ref
        x2, y2 = point
        return math.hypot(x2 - x1, y2 - y1)
    
    @staticmethod
    def merge_clusters(grouped_points, group_cluster_centers, frame_id):
        max_leg_radius_in_mm = 150

        new_grouped_points = []
        new_cluster_centers = []

        for current_max_distance in [100, 150, 200, 250, 300, 350, 400]:
            grouped_clusters, _ = LidarClustering.group_points(group_cluster_centers, current_max_distance)
            if len(grouped_clusters) < 3:
                break

        merged_groups_xs = np.concatenate([grouped_points[k][0] for k in range(len(grouped_points))])
        merged_groups_ys = np.concatenate([grouped_points[k][1] for k in range(len(grouped_points))])

        X = np.array(list(zip(*[merged_groups_xs, merged_groups_ys])))

        if len(grouped_clusters) == 2:
            #print("group1", grouped_clusters[0])
            #print("group2", grouped_clusters[1])
            #print("groups", grouped_clusters)
            cluster_centers = [np.mean(np.array(clusters), axis=0) for clusters in grouped_clusters]

        else:
            #print(f"Using k-means fallback {frame_id}")
            k_means = KMeans(init='k-means++', n_clusters=2, n_init=1)
            k_means.fit(X)
            k_means_labels = k_means.labels_
            cluster_centers = k_means.cluster_centers_
            k_means_labels_unique = np.unique(k_means_labels)



        points_in_group1 = X[np.where(np.linalg.norm(X-cluster_centers[0], axis=1) < max_leg_radius_in_mm)]
        points_in_group2 = X[np.where(np.linalg.norm(X-cluster_centers[1], axis=1) < max_leg_radius_in_mm)]

        xs_g1, ys_g1 = list(zip(*points_in_group1))
        xs_g2, ys_g2 = list(zip(*points_in_group2))

        new_grouped_points.append([xs_g1, ys_g1])
        new_grouped_points.append([xs_g2, ys_g2])

        new_cluster_centers.append([np.mean(xs_g1), np.mean(ys_g1)])
        new_cluster_centers.append([np.mean(xs_g2), np.mean(ys_g2)])

        return new_grouped_points, new_cluster_centers, 2
    
    @staticmethod
    def group_points_in_frames(df_merged, verbose=True):
        grouped_points_per_frame = []
        group_cluster_centers_per_frame = []

        more_than_two = []
        for frame_id in range(len(df_merged)):
            grouped_points, group_cluster_centers, cluster_count = LidarClustering.group_motions_in_frame(df_merged, frame_id=frame_id)

            if cluster_count > 2:
                #print(f"More than 2 clusters, fixing Frame {frame_id}")
                grouped_points, group_cluster_centers, cluster_count = LidarClustering.merge_clusters(
                    grouped_points, group_cluster_centers, frame_id
                )

            elif frame_id > 0 and cluster_count == 1: # and (len(group_cluster_centers_per_frame[frame_id-1]) == 2):
                #print(f"Splitting Frame {frame_id} into two clusters")
                grouped_points, group_cluster_centers, cluster_count = LidarClustering.split_cluster_into_two(grouped_points)

            more_than_two.append(cluster_count)
            grouped_points_per_frame.append(grouped_points)
            group_cluster_centers_per_frame.append(group_cluster_centers)
            
            if verbose:
                print(f"Clustered Frame {frame_id + 1} / {len(df_merged)} with a cluster count of {cluster_count}")
        
        return grouped_points_per_frame, group_cluster_centers_per_frame, more_than_two
    
    
class LidarFeet:
  
    @staticmethod
    def assign_clusters_to_feet(group_cluster_centers_per_frame):
        start_indices = []
        end_indices = []
        matches = []
        left_leg = defaultdict(lambda: None)
        rigth_leg = defaultdict(lambda: None)
        walking_directions_left_leg = defaultdict(lambda: None)
        walking_directions_right_leg = defaultdict(lambda: None)

        for frame_id, (left_group, right_group) in enumerate(zip(group_cluster_centers_per_frame, group_cluster_centers_per_frame[1:])):    
            if len(left_group) == 0 and len(right_group) > 0:
                start_indices.append(frame_id)

                #print(f"start found at frame {frame_id}")
            elif len(left_group) > 0 and len(right_group) == 0:
                end_indices.append(frame_id)
                #print(f"end found at frame {frame_id}")
            elif len(left_group) == 2 and len(right_group) == 2:
                distances = np.linalg.norm(np.array(right_group) - left_group[0], axis=1)
                index_of_clostest_dist = np.where(distances == np.min(distances))[0][0]

                match1 = [left_group[0], right_group[index_of_clostest_dist]]
                match2 = [left_group[1], right_group[1 - index_of_clostest_dist]]

                a = np.array(match1[1]) - np.array(match1[0])
                an = a / np.linalg.norm(a)

                b = np.array(match2[1]) - np.array(match2[0])
                bn = b / np.linalg.norm(b)

                walking_direction = an + bn
                walking_direction = walking_direction / np.linalg.norm(walking_direction)

                feet_y = [left_group[0][1], left_group[1][1]]
                top_feet_y_index = np.where(feet_y == np.max(feet_y))[0][0]

                #if frame_id == 3504 or frame_id == 3505 or frame_id == 3506:
                #    print(f"Frame {frame_id} {walking_direction} => {walking_direction[0] < 0}")

                # TODO: instead of checking sign, compute vector between clusters (pointing to top foot. dotprot(topfoot_vec, dir) negative => right to left, otherwise left to right
                if walking_direction[0] < 0:
                    # top y belongs to right foot
                    # bot y belongs to left foot

                    rigth_leg[frame_id] = left_group[top_feet_y_index]
                    left_leg[frame_id] = left_group[1 - top_feet_y_index]

                else:
                    # top y belongs to left foot
                    # bot y belongs to right foot

                    rigth_leg[frame_id] = left_group[1 - top_feet_y_index]
                    left_leg[frame_id] = left_group[top_feet_y_index]

                #print(walking_direction)

                walking_directions_left_leg[frame_id] = a
                walking_directions_right_leg[frame_id] = b
                matches.append([match1, match2])

                # TODO 1. compute best correspondences, 
                # TODO 2. compute direction vector, 
                # TODO 3. assign leg
            else:
                pass

        return left_leg, rigth_leg, start_indices, end_indices, walking_directions_left_leg, walking_directions_right_leg
    
    @staticmethod
    def should_swap_feed_assignments(left_leg, rigth_leg, frame_id):
        lf = np.array(left_leg[frame_id])
        rf = np.array(rigth_leg[frame_id])

        # in these experiments, rf is always above lf and therefore this sanity check holds true
        # TODO: generalize this to utilize walking direction and compute dot pructs.
        if rf[1] > lf[1]:
            return False

        lf_prev = np.array(left_leg[frame_id - 1])
        rf_prev = np.array(rigth_leg[frame_id - 1])

        # original distances
        d1 = np.linalg.norm(lf - lf_prev)
        d2 = np.linalg.norm(rf - rf_prev)

        # swapped distances
        d3 = np.linalg.norm(lf - rf_prev)
        d4 = np.linalg.norm(rf - lf_prev)

        return (d1 + d2) > (d3 + d4)

    @staticmethod
    def correct_feet_clusters(left_leg, rigth_leg, walking_directions_left_leg, walking_directions_right_leg, frame_count):
        right_foot_valid = defaultdict(lambda: True)
        left_foot_valid = defaultdict(lambda: True)

        for frame_id in range(frame_count):
            lf = np.array(left_leg[frame_id])
            rf = np.array(rigth_leg[frame_id])

            lf_prev = np.array(left_leg[frame_id - 1])
            rf_prev = np.array(rigth_leg[frame_id - 1])


            if lf.any() and rf.any() and lf_prev.any() and rf_prev.any():   
                if LidarFeet.should_swap_feed_assignments(left_leg, rigth_leg, frame_id):
                    tmp = left_leg[frame_id]
                    left_leg[frame_id] = rigth_leg[frame_id]
                    rigth_leg[frame_id] = tmp
            else:
                pass
                #print(f"Missing data for frame {frame_id}")
                #print("lf = ", lf)
                #print("rf = ", rf)

        return left_leg, rigth_leg

class DatasetProcessor:
    TASKS = {
        "00_free_walk_1": "bf8a163b-8f55-4885-b3ab-cf8d26f3904c",
        "01_free_heel": "4b53b220-1c96-4af5-a3a5-0d377fd22b2a",
        "06_fast_long": "65142a2d-b26d-45a0-906e-89d63fca32e5",
        "07_normal_long": "f32593da-4168-4684-8653-567186458004",
        "08_slow_long": "f32593da-4168-4684-8653-567186458004", # TODO: ask Aaron
        "13_fga_01": "c6c44aa1-bcfc-4f8d-a451-e42dac3c6a2b" # TODO: ask Aaron
    }
    
    @staticmethod
    def find_start_end_of_sequence(vec, min_interval_length=10):
        # find sequences and lengths
        # seqs = [(key, length), ...]
        seqs = [(key, len(list(val))) for key, val in groupby(vec)]
        # find start positions of sequences
        # seqs = [(key, start, length), ...]
        seqs = [(key, sum(s[1] for s in seqs[:i]), len) for i, (key, len) in enumerate(seqs)]

        #print([[s[1], s[1] + s[2] - 1] for s in seqs if s[0] == 1])
        #print([[s[1], s[1] + s[2] - 1] for s in seqs if s[0] == 1 and s[2] > 2])

        start_ends = [[s[1], s[1] + s[2] - 1] for s in seqs if s[0] == 1 and s[2] > 2]
        filtered_start_ends = list(filter(lambda interval: interval[1] - interval[0] > min_interval_length, start_ends))
        return filtered_start_ends

    @staticmethod
    def draw_walk_interval_pictures(more_than_two, dataset):
        walk_interval_image_filepath = DatasetProcessor.create_walking_interval_filepath(dataset)

        participant = dataset["participant"]
        task = dataset["task"]
        trial_nr = dataset["trial_nr"]

        title_text = f"{participant}_{task}_{trial_nr}"   

        import matplotlib
        from matplotlib import pyplot as plt
        from IPython.display import display, HTML
        matplotlib.use('Agg')

        fig = plt.figure(figsize=(8, 5))
        plt.title(title_text)
        start_endpoints = DatasetProcessor.find_start_end_of_sequence(np.array(more_than_two) > 0)
        plt.plot(np.array(more_than_two) > 0, ".-")
        for pse in start_endpoints:
            plt.text(pse[0], 1.01, str(pse[0]), color="red", fontsize=5)
            plt.text(pse[1], 1.01, str(pse[1]), color="red", fontsize=5)

        plt.savefig(walk_interval_image_filepath)
        plt.close(fig)
    
    @staticmethod
    def create_experiment_json(base_dir_name="gailo"):
        experiments = []

        def participant_number(item):
            splits = item.split("_")
            if len(splits) == 1:
                return 0
            return int(splits[1])

        patient_directories = list(os.walk(base_dir_name))[0][1]
        patient_directories = list(sorted(patient_directories, key=lambda item: participant_number(item)))
        print(patient_directories)
        for participant in patient_directories:
            trial_tasks = set()
            files_in_dir = list(os.walk(f"{base_dir_name}/{participant}"))[0][2]
            for file_in_dir in files_in_dir:
                _, _, _, task, trial_nr, *rest = file_in_dir.split("_")
                trial_tasks.add(f"{task}_{trial_nr}")

            for trial_task in list(trial_tasks):
                task, trial_nr = trial_task.split("_")
                experiments.append({
                    "base_path": base_dir_name,
                    "participant": participant,
                    "task": task,
                    "trial_nr": trial_nr,
                    "walk_range": []
                })

        with open(f"{base_dir_name}_experiments.json", 'w') as f:
            json.dump(experiments, f, indent=2)

    @staticmethod        
    def load_dataset(dataset):
        base_path = dataset["base_path"]
        participant = dataset["participant"]
        task = dataset["task"]
        trial_nr = dataset["trial_nr"]

        df1 = pd.read_parquet(f"{base_path}/{participant}/df_{participant}_{task}_{trial_nr}_200.parquet", engine='pyarrow')
        df2 = pd.read_parquet(f"{base_path}/{participant}/df_{participant}_{task}_{trial_nr}_201.parquet", engine='pyarrow')
        df3 = pd.read_parquet(f"{base_path}/{participant}/df_{participant}_{task}_{trial_nr}_203.parquet", engine='pyarrow')

        dfs = [df1, df2, df3]

        return dfs
    
    @staticmethod
    def leg_dataset_filepath(dataset):
        base_path = dataset["base_path"]
        participant = dataset["participant"]
        task = dataset["task"]
        trial_nr = dataset["trial_nr"]

        filename = f"df_{participant}_{task}_{trial_nr}_with_legs.parquet"
        return os.path.join(base_path, participant, filename)

    
    @staticmethod
    def save_leg_df(df_merged, left_leg, rigth_leg, grouped_points, grouped_cluster_centers, more_than_two, output_filepath):
        lf_x = [None for _ in range(len(df_merged))]
        lf_y = [None for _ in range(len(df_merged))]

        rf_x = [None for _ in range(len(df_merged))]
        rf_y = [None for _ in range(len(df_merged))]


        for idx, centroid in left_leg.items():
            if centroid:
                lf_x[idx] = centroid[0]
                lf_y[idx] = centroid[1]

        for idx, centroid in rigth_leg.items():
            if centroid:
                rf_x[idx] = centroid[0]
                rf_y[idx] = centroid[1]

        df_merged["lf_x"] = lf_x
        df_merged["lf_y"] = lf_y
        df_merged["rf_x"] = rf_x
        df_merged["rf_y"] = rf_y
        df_merged["grouped_points"] = grouped_points
        df_merged["grouped_cluster_centers"] = grouped_cluster_centers
        df_merged["cluster_count"] = more_than_two

        df_merged.to_parquet(output_filepath)
  
    @staticmethod
    def create_leg_fileapth(dataset):
        base_path = dataset["base_path"]
        participant = dataset["participant"]
        task = dataset["task"]
        trial_nr = dataset["trial_nr"]

        return f"{base_path}/{participant}/df_{participant}_{task}_{trial_nr}_with_legs.parquet"
    
    @staticmethod
    def create_walking_interval_filepath(dataset):
        participant = dataset["participant"]
        task = dataset["task"]
        trial_nr = dataset["trial_nr"]

        return f"debug/{participant}_{task}_{trial_nr}.jpeg"

    @staticmethod
    def select_datasets_by_task(datasets, task: str):
        return list(filter(lambda dataset: dataset["task"] == task, datasets))    


    @staticmethod
    def participants_with_all_for(datasets):
        t01 = DatasetProcessor.select_datasets_by_task(datasets, TASKS["00_free_walk_1"])
        t02 = DatasetProcessor.select_datasets_by_task(datasets, TASKS["01_free_heel"])
        t03 = DatasetProcessor.select_datasets_by_task(datasets, TASKS["06_fast_long"])
        t04 = DatasetProcessor.select_datasets_by_task(datasets, TASKS["07_normal_long"])

        project_participants = lambda ds: set([d["participant"] for d in ds])

        pt01 =  project_participants(t01)
        pt02 =  project_participants(t02)
        pt03 =  project_participants(t03)
        pt04 =  project_participants(t04)

        z01 = pt01.intersection(pt02)
        z02 = z01.intersection(pt03)

        return z02.intersection(pt04)
    
    @staticmethod
    def compute_clusters(datasets):
        clustering_errors = []
        grouping_data = {}
        for dataset_idx, dataset in enumerate(datasets):
            try:
                print(f"Processing dataset {dataset} {dataset_idx + 1} / {len(datasets)} ...")
                dfs = DatasetProcessor.load_dataset(dataset)
                df_merged = LidarPreprocessor.merge_lidar_dataframes(dfs)
                grouped_points_per_frame, group_cluster_centers_per_frame, more_than_two = LidarClustering.group_points_in_frames(df_merged)
                grouping_data[dataset_idx] = [grouped_points_per_frame, group_cluster_centers_per_frame, more_than_two]
            except Exception as err:
                print(err)
                clustering_errors.append(dataset)
                print("Error for dataset: ", dataset)

        return grouping_data, clustering_errors

    @staticmethod
    def compute_legs_from(datasets, grouping_data):
        error_datasets = []
        for dataset_idx, dataset in enumerate(datasets):
            try:
                print(f"Processing leg data for dataset {dataset} {dataset_idx + 1} / {len(datasets)} ...")
                dfs = DatasetProcessor.load_dataset(dataset)
                df_merged = LidarPreprocessor.merge_lidar_dataframes(dfs)
                grouped_points_per_frame, group_cluster_centers_per_frame, more_than_two = grouping_data[dataset_idx]

                left_leg, rigth_leg, start_indices, end_indices, walking_directions_left_leg, walking_directions_right_leg = LidarFeet.assign_clusters_to_feet(group_cluster_centers_per_frame)
                left_leg, rigth_leg = LidarFeet.correct_feet_clusters(
                    left_leg, rigth_leg, walking_directions_left_leg, walking_directions_right_leg, len(group_cluster_centers_per_frame)
                )

                output_filepath = DatasetProcessor.create_leg_fileapth(dataset)
                DatasetProcessor.save_leg_df(
                    df_merged, 
                    left_leg, 
                    rigth_leg, 
                    grouped_points_per_frame, 
                    group_cluster_centers_per_frame, 
                    more_than_two, 
                    output_filepath
                )
                DatasetProcessor.draw_walk_interval_pictures(more_than_two, dataset)
                print(f"Saved leg data to {output_filepath}")
            except Exception as err:
                error_datasets.append(dataset)
                print(err)
                print("Error for dataset: ", dataset)

        return error_datasets
    
    @staticmethod
    def preprocess_walks(datasets):
        error_datasets = []
        dfs_list = []
        for dataset_idx, dataset in enumerate(datasets):
            try:
                print(f"Processing leg data for dataset {dataset} {dataset_idx + 1} / {len(datasets)} ...")
                dfs = LidarPreprocessor.preprocess_walk(dataset, save_df=True)
                dfs_list.append(dfs)
            except Exception as err:
                error_datasets.append(dataset)
                print(err)
                print("Error for dataset: ", dataset)
        return dfs_list, error_datasets
    
class LidarGaitParameters:

    @staticmethod
    def interpolate_feet_coordinates(df):
        "this function mutates the provided dataframe"


        def interpolate(timestamps, y_values, order=5):
            filter_values = list(filter(lambda item: np.isnan(item[1]) == False, list(zip(timestamps, y_values))))
            timestamp, lf_x = list(zip(*filter_values))

            return make_interp_spline(timestamp, lf_x, k=order)


        timestamps = np.array(df.timestamp)

        lf_x_spline = interpolate(timestamps, np.array(df.lf_x))
        lf_y_spline = interpolate(timestamps, np.array(df.lf_y))
        rf_x_spline = interpolate(timestamps, np.array(df.rf_x))
        rf_y_spline = interpolate(timestamps, np.array(df.rf_y))

        corrected_lf_x = lf_x_spline(df.timestamp)
        corrected_lf_y = lf_y_spline(df.timestamp)
        corrected_rf_x = rf_x_spline(df.timestamp)
        corrected_rf_y = rf_y_spline(df.timestamp)

        df["lf_x"] = corrected_lf_x
        df["lf_y"] = corrected_lf_y
        df["rf_x"] = corrected_rf_x
        df["rf_y"] = corrected_rf_y

        return df

    @staticmethod
    def get_velocities(df):
        """ Calculate the instantaneous velocity for both legs

        Input:
        df -- the dataframe containing all localisation and time information
        Outputs:
        r_velocity -- the velocity of the right leg in [m/s]
        l_velocity -- the velocity of the left leg in [m/s]
        """

        df['lf_x_diff'] = df['lf_x'].diff()
        df['lf_y_diff'] = df['lf_y'].diff()
        df['lf_diff'] = df[['lf_x_diff', 'lf_y_diff']].values.tolist()
        df['rf_x_diff'] = df['rf_x'].diff()
        df['rf_y_diff'] = df['rf_y'].diff()
        df['rf_diff'] = df[['rf_x_diff', 'rf_y_diff']].values.tolist()

        r_velocity = []
        l_velocity = []

        dt = df.timestamp.diff() / np.timedelta64(1, 's')

        dist_lf = df['lf_diff'].apply(np.linalg.norm) / 1000
        dist_rf = df['rf_diff'].apply(np.linalg.norm) / 1000

        df['lf_velocity'] = dist_lf.values/dt
        df['rf_velocity'] = dist_rf.values/dt

        return df
    
    @staticmethod
    def get_median_filter_velocities(df, right_window_width=15, left_window_width=15):
        """ Apply a median filter to the velocity profiles

        Input:
        df -- the dataframe containing all localisation and time information
        right_window_width -- the window width of the right leg median filter
        left_window_width -- the window width of the left leg median filter
        Ouputs:
        df -- the dataframe containing all localisation and time information completed with median filtered velocities
        """
        
        df['rf_velocity_medfilt'] = df['rf_velocity'].rolling(window=right_window_width, center=True).median()
        df['lf_velocity_medfilt'] = df['lf_velocity'].rolling(window=left_window_width, center=True).median()

        return df
    
    @staticmethod
    def find_peak_velocities(df, distance=[30, 15], height=[0.4, 0.4], small_median_win=5, large_median_win=15):
        """ Find the peak velocity of the input signals
        Inputs:
        r_signal -- the right side signal
        l_signal -- the left side signal
        distance -- the expected minimal distance between peaks
        height -- the expected minimal peak height
        Outputs:
        r_peaks_indices -- the right side signal peak indices
        l_peaks_indices -- the left side signal peak indices
        """
        
        left_velocity = np.array(df['lf_velocity'].values)
        left_velocity[0]=left_velocity[1]
        y2 = medfilt(left_velocity, small_median_win)
        y3 = medfilt(left_velocity, large_median_win)
        l_peaks_indices2, _ = find_peaks(y2, height=height[0], distance=distance[0])
        l_peaks_indices3, _ = find_peaks(y2, distance=distance[1])

        neg_vel = -y2 + max(y2)
        l_min_indices, _ = find_peaks(neg_vel)#, height=0.1, distance=20)

        browns = l_min_indices
        reds = l_peaks_indices2
        purples =l_peaks_indices3
        purples = np.array([purples[i] for i in range(len(purples)) if purples[i] >= reds[0]])

        if reds[-1] >= purples[-1]:
            reds = reds[:-1]

        def indices_values(indices, left_bound, right_bound):
            selected_indices = np.where((indices < right_bound) & (indices > left_bound))
            if len(selected_indices) == 0:
                print("error: empty index set")
                #print("indices=", indices)
                #print("left_bound=", left_bound)
                #print("right_bound=", right_bound)
            return selected_indices[0]

        def convert_to_index(indices_list):

            median_index = np.median(np.array(indices_list))
            if len(indices_list) == 0:
                return None
            return int(median_index)

        red_len = int(len(purples) / 2.0)
        left_secondary_minimas = [convert_to_index(indices_values(browns, reds[k], purples[2*k+1])) for k in range(red_len)]
        left_secondary_minimas = list(filter(lambda item: item != None, left_secondary_minimas))

        l_min_indices = l_min_indices[left_secondary_minimas]

        right_velocity = np.array(df['rf_velocity'].values)
        right_velocity[0] = right_velocity[1]
        y2 = medfilt(right_velocity, small_median_win)
        y3 = medfilt(right_velocity, large_median_win)
        r_peaks_indices2, _ = find_peaks(y2, height=height[0], distance=distance[0])
        r_peaks_indices3, _ = find_peaks(y2, distance=distance[1])

        neg_vel = -y2 + max(y2)
        r_min_indices, _ = find_peaks(neg_vel)#, height=0.1, distance=20)

        browns = r_min_indices
        reds = r_peaks_indices2
        purples = r_peaks_indices3

        purples = np.array([purples[i] for i in range(len(purples)) if purples[i] >= reds[0]])

        if reds[-1] >= purples[-1]:
            reds = reds[:-1]

        red_len = int(len(purples) / 2.0)
        right_secondary_minimas = [convert_to_index(indices_values(browns, reds[k], purples[2*k+1])) for k in range(red_len)]
        right_secondary_minimas = list(filter(lambda item: item != None, right_secondary_minimas))

        r_min_indices = r_min_indices[right_secondary_minimas]

        return l_min_indices,r_min_indices, l_peaks_indices2, r_peaks_indices2
    
    @staticmethod
    def get_cadence(df, l_min_indices,r_min_indices, right_range=[0, -1], left_range=[0, -1]):
        """ Compute cadence based on identified steps

        Inputs:
        r_step_indices -- indices of identified right leg steps as a ndarray
        l_step_indices -- indices of identified left leg steps as a ndarray
        right_range -- limit elements to consider in r_step_indices (given as a list of [first_right_leg_idx, last_right_leg_idx])
        left_range -- limit elements to consider in l_step_indices (given as a list of [first_left_leg_idx, last_left_leg_idx])
        Output:
        r_cadence -- the right leg cadence in (1/s)
        l_cadence -- the left leg cadence in (1/s)
        gait_cadence -- both legs cadence in (1/s)
        """
        min_step_cnt = 3 # TODO: determine the ideal minimal detect peaks number for a valid cadence result

        # limit the scope according to given min/max indices to consider
        r_step_indices = r_min_indices
        l_step_indices = l_min_indices


        # Determine right leg cadence on the given window
        r_delta_t = (df.timestamp[r_step_indices[-1]] - df.timestamp[r_step_indices[0]]) / np.timedelta64(1, 's')
        r_step_cnt = len(r_step_indices)
        if (r_step_cnt > min_step_cnt):
            r_cadence = r_step_cnt / r_delta_t
        else:
            r_cadence = -1

        # Determine left leg cadence on the given window
        l_delta_t = (df.timestamp[l_step_indices[-1]] - df.timestamp[l_step_indices[0]]) / np.timedelta64(1, 's')
        l_step_cnt = len(l_step_indices)
        if (l_step_cnt > min_step_cnt):
            l_cadence = l_step_cnt / l_delta_t
        else:
            l_cadence = -1

        # Determine total cadence
        gait_delta_t = (max(df.timestamp[l_step_indices[-1]], 
                            df.timestamp[r_step_indices[-1]]) - min(df.timestamp[l_step_indices[0]],
                                                                    df.timestamp[r_step_indices[0]])
                       ) / np.timedelta64(1, 's')

        gait_step_cnt = len(r_step_indices) + len(l_step_indices)
        if r_cadence == -1 or l_cadence == -1:
            gait_cadence = -1
        else:
            gait_cadence = gait_step_cnt / gait_delta_t

        return r_cadence, l_cadence, gait_cadence
    
    @staticmethod
    def get_stride_length_time(df, l_min_indices,r_min_indices, right_range=[0, -1], left_range=[0, -1]):
        """ Calculate stride length and stride time based on identified step positions
        Stride length is defined as the length of the vector between two stepping positions of the
        same foot.

        Inputs:

        right_range -- limit elements to consider in r_step_indices (given as a list of [first_right_leg_idx, last_right_leg_idx])
        left_range -- limit elements to consider in l_step_indices (given as a list of [first_left_leg_idx, last_left_leg_idx])
        Outputs:
        r_stride_length list -- a list of right leg stride lengths in (mm)
        l_stride_length_list -- a list of left leg stride lengths in (mm)
        r_stride_time_list -- a list of the right stride durations in (s)
        l_stride_time_list -- a list of the left stride durations in (s)
        """
        
        # TODO: simplify method to make it leg unspecific (i.e. return only stride_length_list and avg_stride_length)
        r_step_indices = r_min_indices
        l_step_indices = l_min_indices
        
        # List stride length
        r_stride_length_list = []
        l_stride_length_list = []

        # List stride time
        r_stride_time_list = []
        l_stride_time_list = []

        # limit the scope according to given min/max indices to consider
        r_step_indices = r_step_indices[right_range[0]:right_range[1]]
        l_step_indices = l_step_indices[left_range[0]:left_range[1]]

        # Extract right leg stride lengths
        for i, r_step_idx in enumerate(r_step_indices): 
            if i > 0:
                rf_diff_x = df['rf_x'][r_step_idx] - df['rf_x'][prev_r_step_idx]
                rf_diff_y = df['rf_y'][r_step_idx] - df['rf_y'][prev_r_step_idx]
                rf_diff = [rf_diff_x, rf_diff_y]
                r_stride_length_list.append(np.linalg.norm(rf_diff))
                r_stride_time_list.append((df.timestamp[r_step_idx] - df.timestamp[prev_r_step_idx]) / np.timedelta64(1, 's'))
         
            prev_r_step_idx = r_step_idx

        # Extract left leg stride lengths
        for i, l_step_idx in enumerate(l_step_indices):
            if i > 0:
                lf_diff_x = df['lf_x'][l_step_idx] - df['lf_x'][prev_l_step_idx]
                lf_diff_y = df['lf_y'][l_step_idx] - df['lf_y'][prev_l_step_idx]
                lf_diff = [lf_diff_x, lf_diff_y]
                l_stride_length_list.append(np.linalg.norm(lf_diff))
                l_stride_time_list.append((df.timestamp[l_step_idx] - df.timestamp[prev_l_step_idx]) / np.timedelta64(1, 's'))
            prev_l_step_idx = l_step_idx

        # Compute average stride lengths
        r_avg_stride_length = np.nanmean(r_stride_length_list)
        l_avg_stride_length = np.nanmean(l_stride_length_list)    

        return r_stride_length_list, l_stride_length_list, r_stride_time_list, l_stride_time_list, r_avg_stride_length, l_avg_stride_length
    
    @staticmethod
    def get_step_length_time(df,l_min_indices,r_min_indices, right_range=[0, -1], left_range=[0, -1]):
        """ Compute step length from identified step indices of both legs.
        Step length is defined as the component parallel to the direction of motion of the vector
        between two opposite stepping position (left/right or right/left).

        Inputs:
        r_step_indices -- indices of identified right leg steps as a ndarray
        l_step_indices -- indices of identified left leg steps as a ndarray
        right_range -- limit elements to consider in r_step_indices (given as a list of [first_right_leg_idx, last_right_leg_idx])
        left_range -- limit elements to consider in l_step_indices (given as a list of [first_left_leg_idx, last_left_leg_idx])
        Outputs:
        step_length_list -- list of identified step lengths in (mm)
        r_step_length_list -- list of the right step lengths in (mm)
        l_step_length_list -- list of the left step lengths in (mm)
        r_step_time_list -- list of the right step times in (s)
        l_step_time_list -- list of the left step times in (s)
        """
        
        r_step_indices = r_min_indices
        l_step_indices = l_min_indices

        # Initialize lists
        step_length_list = []
        r_step_length_list = []
        l_step_length_list = []

        step_time_list = []
        r_step_time_list = []
        l_step_time_list = []

        # limit the scope according to given min/max indices to consider
        r_step_indices = r_step_indices[right_range[0]:right_range[1]]
        l_step_indices = l_step_indices[left_range[0]:left_range[1]]

        # Find very first step
        prev_leg = np.argmin([r_step_indices[0], l_step_indices[0]]) # 0 for right leg, 1 for left leg

        # Loop over identified step indices. Step indices are consumed from the beginning of
        # the list. Loop ends when one of the list has been completly consumed.
        while(len(r_step_indices) and len(l_step_indices)):

            # Define current indices
            r_idx = r_step_indices[0]
            l_idx = l_step_indices[0]

            # Find most recent step at the beginning of both lists
            curr_leg = np.argmax([df.timestamp[r_idx], df.timestamp[l_idx]]) # 0 for right leg, 1 for left leg

            # Compute step length and extract step time
            # TODO: make it dependent on the walking direction, and not on the x-direction only
            diff_x = df['rf_x'][r_idx] - df['lf_x'][l_idx]
            diff_y = df['rf_y'][r_idx] - df['lf_y'][l_idx]
            step_length = np.linalg.norm([diff_x, diff_y])
            r_time = (df.timestamp[r_idx] - df.timestamp[l_idx]) / np.timedelta64(1, 's')
            l_time = (df.timestamp[l_idx] - df.timestamp[r_idx]) / np.timedelta64(1, 's')
            step_time = r_time if curr_leg==0 else l_time

            # Check whether we are missing a step. If (not missing) then append step length. If (missing) then overwrite last step length.
            if curr_leg != prev_leg:
                step_length_list.append(step_length)
                step_time_list.append(step_time)
            else:
                step_length_list.insert(-1, step_length)
                step_time_list.insert(-1, step_time)

            # Remove consumed indices at the beginning of the list
            if curr_leg == 0: # if curr_leg is right leg
                l_step_length_list.append(step_length_list[-1])
                l_step_time_list.append((df.timestamp[r_idx]-df.timestamp[l_idx]) / np.timedelta64(1, 's'))
                l_step_indices = np.delete(l_step_indices, 0)
            else:
                r_step_length_list.append(step_length_list[-1])
                r_step_time_list.append((df.timestamp[l_idx] - df.timestamp[r_idx]) / np.timedelta64(1, 's'))
                r_step_indices = np.delete(r_step_indices, 0)

            prev_leg = curr_leg

        # Compute average
        avg_step_length = np.nanmean(step_length_list)
        avg_r_step_length = np.nanmean(r_step_length_list)
        avg_l_step_length = np.nanmean(l_step_length_list)

        return step_length_list, r_step_length_list, l_step_length_list, r_step_time_list, l_step_time_list, round(avg_step_length, 3)

    @staticmethod
    def calculate_gait_parameters(filepath: str, 
                                  walk_idx: list, 
                                  do_filter=False, 
                                  distance=[30, 15], 
                                  height=[0.3, 0.3], 
                                  verbose=True):

        df = pd.read_parquet(filepath)
        df = df.iloc[walk_idx[0]:walk_idx[1]]
        df2 = df.resample('0.025S', on='timestamp').median()
        df2['timestamp']=df2.index
        df2['index']=list(range(len(df2)))
        df2 = df2.set_index('index')

        df2 = LidarGaitParameters.interpolate_feet_coordinates(df2)


        if do_filter:
            rf_x = np.array(df2['rf_x'])
            rf_y = np.array(df2['rf_y'])
            lf_x = np.array(df2['lf_x'])
            lf_y = np.array(df2['lf_y'])

            N = 3
            rf_x = np.convolve(rf_x, np.ones(N) / N, mode='same')
            rf_y = np.convolve(rf_y, np.ones(N) / N, mode='same')
            lf_x = np.convolve(lf_x, np.ones(N) / N, mode='same')
            lf_y = np.convolve(lf_y, np.ones(N) / N, mode='same')

            rf_x[0] = df2['rf_x'][0]
            rf_y[0] = df2['rf_y'][0]
            lf_x[0] = df2['lf_x'][0]
            lf_y[0] = df2['lf_y'][0]


            rf_x[-1] = df2['rf_x'][len(df2)-1]
            rf_y[-1] = df2['rf_y'][len(df2)-1]
            lf_x[-1] = df2['lf_x'][len(df2)-1]
            lf_y[-1] = df2['lf_y'][len(df2)-1]


            df2['rf_x'] = rf_x
            df2['rf_y'] = rf_y
            df2['lf_x'] = lf_x
            df2['lf_y'] = lf_y

        df2 = LidarGaitParameters.get_velocities(df2)
        df2 = LidarGaitParameters.get_median_filter_velocities(df2,right_window_width=15, left_window_width=15)
       
        l_min_indices,r_min_indices, l_peaks_indices, r_peaks_indices = LidarGaitParameters.find_peak_velocities(df2, distance=distance, height=height)
        r_cadence, l_cadence, gait_cadence = LidarGaitParameters.get_cadence(df2, l_min_indices,r_min_indices)
        r_stride_length_list, l_stride_length_list, r_stride_time_list, l_stride_time_list, r_avg_stride_length, l_avg_stride_length = LidarGaitParameters.get_stride_length_time(df2, l_min_indices,r_min_indices)

        step_length_list, r_step_length_list, l_step_length_list, r_step_time_list, l_step_time_list, avg_step_len = LidarGaitParameters.get_step_length_time(df2,l_min_indices,r_min_indices)


        step_count = len(r_step_time_list) + len(l_step_time_list)
        total_time_s = np.sum(r_step_time_list) + np.sum(l_step_time_list)
        avg_step_time_s = 0.5 * (np.mean(r_step_time_list) + np.mean(l_step_time_list))
        avg_step_len_mm = np.mean(step_length_list)
        avg_stride_len_cm = (0.5 * (l_avg_stride_length + r_avg_stride_length)) / 10.0
        avg_cadence_per_min = (step_count / total_time_s) * 60
        avg_step_len_cm = avg_step_len_mm / 10.0
        avg_velocity_cm_per_s = avg_step_len_cm / avg_step_time_s

        cycle_time = 0.5 *(np.mean(r_stride_time_list) + np.mean(l_stride_time_list))


        result = {
            "step_count": step_count,
            "ambulation_time": total_time_s,
            "avg_step_time": avg_step_time_s,
            "avg_step_length": avg_step_len_cm,
            "avg_stride_length": avg_stride_len_cm,
            "avg_velocity": avg_velocity_cm_per_s,
            "avg_cadence": avg_cadence_per_min,
            "avg_cycle_time": cycle_time
        }

        if verbose:
            print("Average stride length (cm): ", f"{avg_stride_len_cm} [L={l_avg_stride_length / 10.0} R={r_avg_stride_length / 10.0}]")
            print("Average step time (s): ", avg_step_time_s)
            print("Average step length (cm): ", avg_step_len_cm)
            print("Average Velocity (cm / s): ", avg_velocity_cm_per_s)
            print("Step Count: ", step_count)
            print("Cadence (steps / min): ", avg_cadence_per_min)
            print("Cycle Time (s): ", cycle_time)
            
        return df2, result
    
    @staticmethod
    def evaluate_dataset(datasets, output_filename="gailo", save_results=True):
        results = []
        for dataset_idx, configuration in enumerate(datasets):
            print(f"Evaluating dataset {dataset_idx + 1} / {len(datasets)}...")
            filepath = configuration["dataset"]
            walk_interval = configuration["range"]
            do_filter = "do_filter" in configuration
            distance = [30, 15]
            if "distance" in configuration:
                distance = configuration["distance"]

            _, result = LidarGaitParameters.calculate_gait_parameters(
                filepath, walk_interval, do_filter=do_filter, distance=distance, verbose=False
            )
            result["dataset"] = filepath
            results.append(result)
            
        with open(f"{output_filename}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
    