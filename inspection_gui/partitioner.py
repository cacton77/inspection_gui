import time
import threading
import math
import copy
import numpy as np
import open3d as o3d
from math import pi
from sklearn.cluster import KMeans  # . . . . . . . . K-means
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale
from open3d.geometry import PointCloud, TriangleMesh
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction


class Partitioner:
    def __init__(self, ):

        self.fov_height = 20.0
        self.fov_width = 30.0
        self.dof = 20.0

        self.k = 1
        self.point_weight = 1.0
        self.lambda_weight = 1.0
        self.beta_weight = 1.0
        self.normal_weight = 1.0
        self.initial_method = 'random'
        self.maximum_iterations = 100
        self.number_of_runs = 10
        self.bs_high_scaling = 2
        self.k_evaluation_tries = 1
        self.area_per_point = 0
        self.eval_tries = 0
        self.min_tires = 0
        self.pcd_common = PointCloud()
        self.ppsqmm = 10
        self.valid_pcds = []
        self.overall_packing_efficiency = 0
        self.total_point_out_percentage = 0
        self.total_planar_pcds = 0

    def evaluate_cluster(self, pcd):
        """ Function to be implemented for testing different ways of evaluating validity of a cluster. """
        obb = pcd.get_minimal_oriented_bounding_box(robust=True)
        t = obb.get_center()
        R = obb.R

        if not pcd.has_normals():
            pcd.estimate_normals()
        pcd.normalize_normals()

        normals = np.asarray(pcd.normals)
        z = np.array([0, 0, 1])
        z_hat = np.average(normals, 0)
        x_hat = np.cross(z, z_hat)
        y_hat = np.cross(z_hat, x_hat)

        x_hat = x_hat/np.linalg.norm(x_hat)
        y_hat = y_hat/np.linalg.norm(y_hat)
        z_hat = z_hat/np.linalg.norm(z_hat)

        R = np.hstack(
            (x_hat.reshape(3, 1), y_hat.reshape(3, 1), z_hat.reshape(3, 1)))

        pcd.rotate(np.linalg.inv(R), pcd.get_center())
        pcd.translate(-pcd.get_center())
        pcd.estimate_normals()
        pcd.normalize_normals()
        # temp_pcd_hull, _ = pcd.compute_convex_hull()
        obb = pcd.get_minimal_oriented_bounding_box(robust=True)
        obb.color = (0, 1, 0)

        camera_width = self.fov_width
        camera_height = self.fov_height
        camera_r = camera_height/2
        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.zeros((len(pcd_points), 3))

        valid = True

        try:
            red_count = 0
            green_count = 0
            x_max = 0
            x_min = 0
            y_max = 0
            y_min = 0
            for i in range(len(pcd_points)):
                p = pcd_points[i, :]
                x, y, z = p[0], p[1], p[2]
                x_max = x if x > x_max else x_max
                x_min = x if x < x_min else x_min
                y_max = y if y > y_max else y_max
                y_min = y if y < y_min else y_min
                if math.sqrt(x**2 + y**2) > camera_r or abs(z) > self.dof/2:
                    pcd_colors[i, 0] = 1.  # paint red
                    red_count += 1
                else:
                    pcd_colors[i, 1] = 1.  # paint green
                    green_count += 1
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
        except Exception as e:
            print(e)
        if (red_count/(red_count+green_count) < 0.20):
            valid = True
        else:
            valid = False
        cost = (x_max - x_min)*(y_max - y_min)/(pi*camera_r**2)
        return valid, cost

    def evaluate_cluster_dof(self, pcd):
        """ Function for testing different ways of evaluating validity of a cluster. """

        if not pcd.has_normals():
            pcd.estimate_normals()
        pcd.normalize_normals()

        normals = np.asarray(pcd.normals)
        z = np.array([0, 0, 1])
        z_hat = np.average(normals, 0)
        x_hat = np.cross(z, z_hat)
        y_hat = np.cross(z_hat, x_hat)

        x_hat = x_hat/np.linalg.norm(x_hat)
        y_hat = y_hat/np.linalg.norm(y_hat)
        z_hat = z_hat/np.linalg.norm(z_hat)

        R = np.hstack(
            (x_hat.reshape(3, 1), y_hat.reshape(3, 1), z_hat.reshape(3, 1)))

        pcd.rotate(np.linalg.inv(R), pcd.get_center())
        pcd.translate(-pcd.get_center())
        pcd.estimate_normals()
        pcd.normalize_normals()

        camera_width = self.fov_width
        camera_height = self.fov_height
        camera_r = camera_height/2
        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.zeros((len(pcd_points), 3))
        valid = True

        try:
            red_count = 0
            green_count = 0
            z_max = 0
            z_min = 0
            for i in range(len(pcd_points)):
                p = pcd_points[i, :]
                x, y, z = p[0], p[1], p[2]
                z_max = z if z > z_max else z_max
                z_min = z if z < z_min else z_min
                if abs(z) > self.dof/2:
                    pcd_colors[i, 0] = 1.  # paint red
                    red_count += 1

                else:
                    pcd_colors[i, 1] = 1.  # paint green
                    green_count += 1
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
        except Exception as e:
            print(e)
        if (red_count/(red_count+green_count) < 0.050):
            valid = True
        else:
            valid = False

        # cost = (x_max - x_min)*(y_max - y_min)

        point_out_percentage = red_count/(green_count+red_count)
        max_height = self.dof
        point_height = abs(z_max-z_min)
        packing_eff = point_height/max_height

        # return valid, cost, green_count, objs
        return valid, point_out_percentage, packing_eff

    def evaluate_cluster_fov(self, pcd):
        """ Function to be implemented for testing different ways of evaluating validity of a cluster. """
        if not pcd.has_normals():
            pcd.estimate_normals()
        pcd.normalize_normals()

        normals = np.asarray(pcd.normals)
        z = np.array([0, 0, 1])
        z_hat = np.average(normals, 0)
        # if(np.linalg.norm(z_hat==0)):
        #     print("zhat is zero")
        x_hat = np.cross(z, z_hat)
        if (np.linalg.norm(x_hat) == 0):
            # print("x_hat is zero")
            # print(z_hat)
            x_hat = np.array([z_hat[2], 0, 0])
            print(x_hat)
        y_hat = np.cross(z_hat, x_hat)
        if (np.linalg.norm(y_hat) == 0):
            # print("y_hat is zero")
            # print(x_hat, z_hat)
            y_hat = np.array([0, 1, 0])
        x_hat = x_hat/np.linalg.norm(x_hat)
        y_hat = y_hat/np.linalg.norm(y_hat)
        z_hat = z_hat/np.linalg.norm(z_hat)

        R = np.hstack(
            (x_hat.reshape(3, 1), y_hat.reshape(3, 1), z_hat.reshape(3, 1)))

        pcd.rotate(np.linalg.inv(R), pcd.get_center())
        pcd.translate(-pcd.get_center())
        pcd.estimate_normals()
        pcd.normalize_normals()
        # temp_pcd_hull, _ = pcd.compute_convex_hull()

        camera_width = self.fov_width
        camera_height = self.fov_height

        camera_r = camera_height/2
        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.zeros((len(pcd_points), 3))

        valid = True

        try:
            red_count = 0
            green_count = 0
            x_max = 0
            x_min = 100000000
            y_max = 0
            y_min = 100000000
            x_max_in = 0
            x_min_in = 100000000
            y_max_in = 0
            y_min_in = 100000000
            extreme_point_length = 0
            for i in range(len(pcd_points)):
                p = pcd_points[i, :]
                x, y, z = p[0], p[1], p[2]
                # x_max = x if x > x_max else x_max
                # x_min = x if x < x_min else x_min
                # y_max = y if y > y_max else y_max
                # y_min = y if y < y_min else y_min
                extreme_point_length = math.sqrt(
                    x**2 + y**2) if math.sqrt(x**2 + y**2) > extreme_point_length else extreme_point_length
                if math.sqrt(x**2 + y**2) > camera_r:
                    pcd_colors[i, 0] = 1.  # paint red
                    red_count += 1

                else:
                    x_max_in = x if x > x_max_in else x_max_in
                    x_min_in = x if x < x_min_in else x_min_in
                    y_max_in = y if y > y_max_in else y_max_in
                    y_min_in = y if y < y_min_in else y_min_in
                    pcd_colors[i, 1] = 1.  # paint green
                    green_count += 1
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
        except Exception as e:
            print(e)
        if self.eval_tries == 0:
            self.area_per_point = 1/self.ppsqmm
            print(self.area_per_point)

            self.eval_tries = 1
        # print(self.area_per_point*(red_count+green_count))

        if (red_count/(red_count+green_count) < 0.003):
            valid = True
        else:
            valid = False
        # cost 1
        # area_in = (x_max_in - x_min_in)*(y_max_in - y_min_in)
        # area_full = (x_max - x_min)*(y_max - y_min)
        # fov_area = pi*(camera_r**2)
        # cost = fov_area-2*area_in+area_full
        # point_out_percentage = red_count/(green_count+red_count)
        # print(f'area={fov_area}: area_in={area_in}: area_full={area_full}')
        # return valid, cost,point_out_percentage, objs
        # cost 1

        # #cost 2
        # area_in = self.area_per_point*green_count
        # area_out = self.area_per_point*red_count
        # fov_area = pi*(camera_r**2)
        # cost = 20*(area_out)+(fov_area-area_in)

        # cost 3
        point_out_percentage = red_count/(green_count+red_count)
        max_points_in = (pi*camera_r**2)/self.area_per_point
        packing_eff = green_count/max_points_in
        borderline = 0
        if (extreme_point_length < (camera_r*1.05) and extreme_point_length > (camera_r*0.95)) or (extreme_point_length < (camera_r*1.05) and extreme_point_length > (camera_r*0.95)) and valid == True:
            borderline = 1
        return valid, point_out_percentage, packing_eff, borderline
        # cost 3

    def partition(self, pcd, k) -> list:
        """ K-Means clustering function. """

        # Unpack and scale vertices and normalize normals from PCD
        if not pcd.has_normals():
            pcd.estimate_normals()

        pcd.normalize_normals()  # Normals must be normalized for correct clustering

        normals = np.asarray(pcd.normals)
        points = np.asarray(pcd.points)

        if (self.normal_weight != 0):
            # Scale point locations to lie between [-1, 1]
            points = 2 * (minmax_scale(points) - 0.5)

        # Combine weighted vertex and normal data
        data = np.concatenate((self.point_weight * points,
                               self.normal_weight * normals), axis=1)
        # data = normals

        # Scikit Learn KMeans
        KM = KMeans(init='k-means++',
                    n_clusters=k,
                    n_init=self.number_of_runs,
                    max_iter=self.maximum_iterations)
        KM.fit(data)

        labels = KM.labels_
        cluster_collection = [[] for i in range(k)]

        for j in range(len(labels)):
            cluster_collection[labels[j]].append(j)

        # List stores regions
        pcds = []
        for i in range(k):
            pcd_i = copy.deepcopy(pcd.select_by_index(cluster_collection[i]))
            pcds.append(pcd_i)

        return pcds

    def evaluate_k(self, pcd, k, eval_fun, tries=1):
        """ Run multiple k-means partitions to determine if current k is valid. """
        for i in range(tries):
            pcds = self.partition(copy.deepcopy(pcd), k)
            k_valid = True
            for j, pcd_1 in enumerate(pcds):
                cluster_valid, cost, _ = eval_fun(
                    copy.deepcopy(pcd_1))
                k_valid = k_valid and cluster_valid
                # print(f'k-{k} pcd {j}: {cluster_valid}')
                if not cluster_valid:
                    break
            if k_valid:
                return True, pcds, cost
        return False, pcds, cost

    def evaluate_k_cost_filter(self, k):
        """ Calls partitioning service to partition surface into planar patches then regions. """
        self.regions = []
        # non_valid_pcd, cost, pcds = self.pcd_partitioner.evaluate_k_cost(copy.deepcopy(
        #     self.pcd.pcd), k, self.camera, self.pcd_partitioner.evaluate_cluster_fov, tries=1)
        cost = self.evaluate_k_cost(copy.deepcopy(
            self.pcd_common), k, self.evaluate_cluster_fov, tries=1)
        return cost

    def evaluate_k_cost(self, pcd, k, eval_fun, tries=1):
        # this is to make sure that k is always an integer and minimum value of k is 1
        k = max(1, int(k))
        for i in range(tries):
            pcds = self.partition(copy.deepcopy(pcd), k)
            k_valid = True
            total_cost = 0
            non_valid_pcd = 0
            total_count = 0
            total_point_out_percentage = 0
            total_packing_eff = 0
            anyborderline = False

            for j, pcd_1 in enumerate(pcds):
                #### cost 1#####
                #     cluster_valid, cost,point_out_percentage,_=eval_fun(copy.deepcopy(pcd_1), camera)
                #     total_cost+=cost
                #     total_point_out_percentage+=point_out_percentage

                # total_point_out_percentage=total_point_out_percentage/(j+1)
                # print(total_point_out_percentage)

                #### cost 2###
                cluster_valid, point_out_percentage, packing_eff, _, borderline = eval_fun(
                    copy.deepcopy(pcd_1))
                total_point_out_percentage += point_out_percentage
                total_packing_eff += packing_eff
                total_count += 1
                # print(f'k-{k} pcd {j}: {cluster_valid}')
                if not cluster_valid:
                    non_valid_pcd += 1
                if (borderline == 1):
                    anyborderline = True
                    # print("border line found")
            total_point_out_percentage = total_point_out_percentage/total_count
            total_packing_eff = total_packing_eff/total_count

            initial_beta = self.beta_weight

            if (total_point_out_percentage > 0.001):
                # print(total_point_out_percentage)
                s = 0
            else:
                print("total_point_out_percentage is zero")
                s = 1
                # print(total_packing_eff,"total packing efficiency in action")
                # self.min_tries+=1
                if (anyborderline == True):
                    s = 0
                # print(total_packing_eff)

            total_cost = (self.lamda_weight)*total_point_out_percentage + \
                s*((1/total_packing_eff)**initial_beta)

        return -total_cost

    def optimize_k(self, pcd, eval_fun, bs_high=1) -> int:
        """ Function to perform K-Means binary search with evaluation to determine optimal number of clusters for inspection """
        print('Finding bs_high...')
        while (True):
            valid, pcd_1, cost = self.evaluate_k(
                copy.deepcopy(pcd), bs_high, eval_fun, tries=1)
            if not valid:
                print(
                    f'bs_high = {bs_high} is not valid, incrementing bs_high...')
                bs_high *= 2
            else:
                print(f'bs_high = {bs_high} is valid. Optimizing k...')
                break
        bs_mid = 0
        bs_low = max(bs_high//2, 1)
        valid_pcds = pcd_1
        while (bs_high > bs_low):
            bs_mid = (bs_low + bs_high)//2
            k = bs_mid
            valid, pcds, _ = self.evaluate_k(
                copy.deepcopy(pcd), k, eval_fun)
            print(f'k: {k}, Valid: {valid}')
            if not valid:
                bs_low = bs_mid + 1
            else:
                valid_pcds = pcds
                bs_high = bs_mid

        k = bs_high

        return k, valid_pcds

    def optimize_k_b_opt(self, pcd, eval_fun, bs_high=1) -> int:
        try:
            self.min_tries = 0
            self.eval_tries = 0
            temp_mesh, _ = pcd.compute_convex_hull(joggle_inputs=True)
            # calculate the bounding box of the mesh
            bounding_box = temp_mesh.get_axis_aligned_bounding_box()
            # calculate the length and width of the bounding box
            length = bounding_box.get_max_bound(
            )[0]-bounding_box.get_min_bound()[0]
            width = bounding_box.get_max_bound(
            )[1]-bounding_box.get_min_bound()[1]
            print(length, width)
            # check if the length is greater than the width
            if length > width:
                greater_dimension = length
                ratio = length/width
            else:
                greater_dimension = width
                ratio = width/length

            camera_width = self.fov_width
            camera_height = self.fov_height
            camera_area = camera_width*camera_height
            camera_r = camera_height/2
            area = temp_mesh.get_surface_area()/2
            print(area)

            if ratio > 10000:
                n_est = greater_dimension/(2*camera_r)
                pbounds = {"k": (n_est/2, 3*(n_est/2))}
                print("case_1")
            else:
                n_est = area/camera_area
                pbounds = {"k": (n_est, 3*n_est)}
                print("case_2")

            print("min area found", n_est)
            # FOV=self.settings.camera_roi_height
            # FOV_area=pi*((FOV/2)**2)
            # upper_bound=2*(area/FOV_area)
            # pbounds={"k":(n_est,2*n_est)}

            # using the bounds as the largest dimension of the planar segment divided by the diameter of the camera which is n_est

            optimizer = BayesianOptimization(
                f=self.evaluate_k_cost_filter,
                pbounds=pbounds,
                verbose=2,  # verbose=1 prints only at max, verbose=0 is silent
                random_state=1,
            )
            acq_function = UtilityFunction(kind="ei", kappa=5)
            optimizer.maximize(
                init_points=1,
                n_iter=3,
            )
            y = optimizer.max
            k = max(1, int(y["params"]["k"]))
            print(k)
            valid_pcds = self.partition(copy.deepcopy(pcd), k)
            non_valid_pcd = 0
            total_count = 0
            total_point_out_percentage = 0
            total_packing_eff = 0

            for j, pcd_1 in enumerate(valid_pcds):
                #### cost 1#####
                #     cluster_valid, cost,point_out_percentage,_=eval_fun(copy.deepcopy(pcd_1), camera)
                #     total_cost+=cost
                #     total_point_out_percentage+=point_out_percentage

                # total_point_out_percentage=total_point_out_percentage/(j+1)
                # print(total_point_out_percentage)

                #### cost 2###
                cluster_valid, point_out_percentage, packing_eff, borderline = eval_fun(
                    copy.deepcopy(pcd_1))
                total_point_out_percentage += point_out_percentage
                total_packing_eff += packing_eff
                total_count += 1
                # print(f'k-{k} pcd {j}: {cluster_valid}')
                if not cluster_valid:
                    non_valid_pcd += 1
            total_point_out_percentage = total_point_out_percentage/total_count
            total_packing_eff = total_packing_eff/total_count
            # print(total_packing_eff,"total packing efficiency for given planar_pcd")
            self.overall_packing_efficiency += total_packing_eff
            self.total_point_out_percentage += total_point_out_percentage
            # print(self.overall_packing_efficiency,"overall packing efficiency")

        #     while (i <= self.settings.partitioning_k):
        #         print(i)
        #         non_valid_pcd, cost = self._app.evaluate_k_cost(i)
        #         k_values.append(i)
        #         cost_value.append(cost)
        #         # self.settings.show_pcd = False
        #         # self._show_pcd.checked = False
        #         # self._apply_settings()
        #         i += 1
        except Exception as e:
            print(e)

        return k, valid_pcds

    def normalsestimation(self, pointcloud, nn_glob, VP=[0, 0, 0]):

        ViewPoint = np.array(VP)
        # datastructure to store normals and curvature
        normals = np.empty(np.shape(pointcloud), dtype=np.float32)
        curv = np.empty((len(pointcloud), 1), dtype=np.float32)

        # loop through the point cloud to estimate normals and curvature
        for index in range(len(pointcloud)):
            # access the points in the vicinity of the current point and store in the nn_loc variable
            nn_loc = pointcloud[nn_glob[index]]
            # calculate the covariance matrix of the points in the vicinity
            COV = np.cov(nn_loc, rowvar=False)
            # calculate the eigenvalues and eigenvectors of the covariance matrix
            eigval, eigvec = np.linalg.eig(COV)
            # sort the eigenvalues in ascending order
            idx = np.argsort(eigval)
            # store the normal of the point in the normals variable
            nor = eigvec[:, idx][:, 0]
            # check if the normal is pointing towards the viewpoint
            if nor.dot((ViewPoint - pointcloud[index, :])) > 0:
                normals[index] = nor
            else:
                normals[index] = -nor
            # store the curvature of the point in the curv variable
            curv[index] = eigval[idx][0] / np.sum(eigval)
        return normals, curv

    # Function to perform region growing on a point cloud

    def regiongrowing1(self, pointcloud, nn_glob, theta_th='auto', cur_th='auto'):

        # Estimate normals and curvature
        # time and print this operation
        start = time.time()
        normals, curvature = self.normalsestimation(
            pointcloud, nn_glob=nn_glob)
        end = time.time()
        print("Time taken to estimate normals and curvature: ", end-start)
        # return a list of indices that would sort the curvature array, pointcloud
        order = curvature[:, 0].argsort().tolist()
        region = []
        cur_th = 'auto'
        # Set default values for theta_th and cur_th
        if theta_th == 'auto':
            theta_th = 15.0 / 180.0 * math.pi  # in radians
        if cur_th == 'auto':
            cur_th = np.percentile(curvature, 98)
        # Perform region growing
        # Loop through the points in the point cloud until all points are assigned to a region
        while len(order) > 0:
            region_cur = []
            seed_cur = []
            # Get the curvature value of the first point of minimum curvature
            poi_min = order[0]
            region_cur.append(poi_min)
            seedval = 0
            # Add the first point index which is the index of the point of minimum curvature to the seed_cur list
            seed_cur.append(poi_min)
            # Remove the index point of minimum curvature from the order list
            order.remove(poi_min)
            # Loop through the seed_cur list until all indexes points in the seed_cur list are assigned to a region
            while seedval < len(seed_cur):
                # Get the nearest neighbors of the current seed point
                nn_loc = nn_glob[seed_cur[seedval]]
                # Loop through the nearest neighbors
                for j in range(len(nn_loc)):
                    # Get the current nearest neighbor index looped through the list of nearest neighbors
                    nn_cur = nn_loc[j]
                    if nn_cur in order:  # Check if nn_cur is in order
                        # find the angle between the normals of the current seed point and the current nearest neighbor
                        dot_product = np.dot(
                            normals[seed_cur[seedval]], normals[nn_cur])
                        angle = np.arccos(np.abs(dot_product))
                        # check for the angle threshold
                        if angle < theta_th:
                            # add the current nearest neighbor to the region_cur list
                            region_cur.append(nn_cur)
                            # remove the current nearest neighbor from the order list
                            order.remove(nn_cur)
                            # check for the curvature threshold
                            if curvature[nn_cur] < cur_th:
                                seed_cur.append(nn_cur)
                # increment the seed value
                seedval += 1
            # append the region_cur list to the region list
            region.append(region_cur)
        # return the region list which contains the indices of the points in each region
        return region

# Region growing added to the smart partition function
    def smart_partition(self):
        """ Partition PCD into Planar Patches, partition Planar Patches into Regions. """
        print(f'Partitioning part into planar patches:')
        # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10,
        #                                                 std_ratio=0.01)
        # pcd= pcd.select_by_index(ind, invert=True)
        self.overall_packing_efficiency = 0
        self.total_point_out_percentage = 0
        theta_th = 4.0 / 180.0 * math.pi  # in radians
        cur_th = 0.01
        num_nieghbors = 30
        pcd = self.pcd
        rg_regions = []
        region_pcds = []
        self.total_planar_pcds = 0

        # store point cloud as numpy array
        unique_rows = np.asarray(pcd.points)
        # Generate a KDTree object
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        search_results = []

        # search for 10 nearest neighbors for each point in the point cloud and store the k value, index of the nearby points and their distances them in search_results
        for point in pcd.points:
            try:
                result = pcd_tree.search_knn_vector_3d(point, num_nieghbors)
                search_results.append(result)
            except RuntimeError as e:
                print(f"An error occurred with point {point}: {e}")
                continue

        # separate the k and index values from the search_results

        k_values = [result[0] for result in search_results]
        nn_glob = [result[1] for result in search_results]
        distances = [result[2] for result in search_results]

        region1 = self.regiongrowing1(unique_rows, nn_glob, theta_th, cur_th)
        # visualize the region with each region having different color
        colors = np.random.rand(len(region1), 3)  # generating random colors
        # initializing the color array
        pcd_colors = np.zeros((len(pcd.points), 3))
        # region stored
        # colour all regions with points less than 1000 with grey and remove them

        initial_normal_weight = self.normal_weight
        for i in range(len(region1)):
            if len(region1[i]) < 0.005*len(unique_rows):
                continue
            else:
                pcd_i = copy.deepcopy(pcd.select_by_index(region1[i]))
                rg_regions.append(pcd_i)
        for j, rg_region in enumerate(rg_regions):
            k_dof, planar_pcds = self.optimize_k(
                copy.deepcopy(rg_region), self.evaluate_cluster_dof)
            region_planar_pcds = []

            self.normal_weight = 0
            for i, planar_pcd in enumerate(planar_pcds):

                print(f'Partitioning planar patch {i} into regions:')
                self.pcd_common = copy.deepcopy(planar_pcd)
                k_roi, pcds = self.optimize_k_b_opt(copy.deepcopy(
                    planar_pcd), self.evaluate_cluster_fov)
                self.total_planar_pcds += 1
                region_planar_pcds += copy.deepcopy(pcds)
            region_pcds.extend(region_planar_pcds)
        self.normal_weight = initial_normal_weight
        total_packing_efficiency = self.overall_packing_efficiency/self.total_planar_pcds
        total_point_out_percentage = self.total_point_out_percentage/self.total_planar_pcds
        print("overall packing efficiency", total_packing_efficiency)
        print("total point out percentage", total_point_out_percentage)
        return region_pcds


# Region growing not added to the smart partition function

    def rg_not_smart_partition(self, pcd) -> list:
        """ Partition PCD into Planar Patches, partition Planar Patches into Regions. """
        print(f'Partitioning part into planar patches:')
        # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10,
        #                                                 std_ratio=0.01)
        # pcd= pcd.select_by_index(ind, invert=True)
        self.overall_packing_efficiency = 0
        self.total_point_out_percentage = 0
        print(self.ppsqmm)
        k_dof, planar_pcds = self.optimize_k(
            copy.deepcopy(pcd), self.evaluate_cluster_dof)
        region_pcds = []
        initial_normal_weight = self.normal_weight
        self.normal_weight = 0
        for i, planar_pcd in enumerate(planar_pcds):

            print(f'Partitioning planar patch {i} into regions:')
            self.pcd_common = copy.deepcopy(planar_pcd)
            k_roi, pcds = self.optimize_k_b_opt(copy.deepcopy(
                planar_pcd), self.evaluate_cluster_fov)

            region_pcds += copy.deepcopy(pcds)
        self.normal_weight = initial_normal_weight
        total_packing_efficiency = self.overall_packing_efficiency / \
            len(planar_pcds)
        total_point_out_percentage = self.total_point_out_percentage / \
            len(planar_pcds)
        print("overall packing efficiency", total_packing_efficiency)
        print("total point out percentage", total_point_out_percentage)
        return region_pcds

    def display_inlier_outlier(self, cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw([inlier_cloud, outlier_cloud])


if __name__ == "__main__":
    partitioner = Partitioner()
    model = o3d.io.read_triangle_mesh(
        "/home/col/Inspection/Parts/2_5D_coupon.stl")
    model.scale(100, [0, 0, 0])
    pcd = model.sample_points_uniformly(number_of_points=100000)
    region_pcds = partitioner.rg_not_smart_partition(pcd)
    # print(f'Number of regions: {len(region_pcds)}')
    # o3d.visualization.draw_geometries(region_pcds)