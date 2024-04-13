// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#include <algorithm>

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2/rsutil.h>


#include <Eigen/Geometry>

#include <pcl/common/angles.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/grid_minimum.h>

#include <pcl/segmentation/min_cut_segmentation.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/search/octree.h>
#include <pcl/range_image/range_image.h>

#include <pcl/visualization/cloud_viewer.h>


using pcl_ptr = pcl::PointCloud<pcl::PointXYZ>::Ptr;
        // a rewritten pcl::SampleConsensusModelPlane::selectWithinDistance that varies the threshold as we move away from the robot
        // in order to compensate for greater noise
        // https://github.com/PointCloudLibrary/pcl/blob/master/sample_consensus/include/pcl/sample_consensus/impl/sac_model_plane.hpp
        void selectOutsideGroundPlane(const pcl_ptr input_cloud, Eigen::Vector4f& plane_coefficients, float threshold, std::vector<int> &inliers)
        {
            int nr_p = 0;
            inliers.resize(input_cloud->size());
            
            float threshold_close    = threshold;
            float threshold_far      = 0.1;
            float threshold_boundary = 1.5 * 1.5; // pre-squared
            
            // Iterate through the 3d points and calculate the distances from them to the plane
            for (size_t i = 0; i < input_cloud->size(); ++i)
            {
              // Calculate the distance from the point to the plane normal as the dot product
              // D = (P-A).N/|N|
              Eigen::Vector4f pt (input_cloud->points[i].x,
                                  input_cloud->points[i].y,
                                  input_cloud->points[i].z,
                                  1);
    
              float distance = fabsf(plane_coefficients.dot(pt));
              
              // check to see whether the point is near or far from us
              float source_distance = std::pow(input_cloud->points[i].x, 2) + std::pow(input_cloud->points[i].y, 2);
              bool near = source_distance < threshold_boundary;
                  
              if ((near && distance < threshold_close)  || (!near && distance < threshold_far)) // (near ? threshold_close : threshold_far))
              {
                // Returns the indices of the points whose distances are smaller than the threshold
                inliers[nr_p] = i;
                ++nr_p;
              }
            }
            inliers.resize(nr_p);
        }
bool remove_groundplane(Eigen::Vector4f& groundplane_model_, const pcl_ptr input_cloud, pcl_ptr output_cloud) 
        {
            //pcl::copyPointCloud(*input_cloud, *output_cloud);
            pcl::SampleConsensusModelPlane<pcl::PointXYZ> plane_model = pcl::SampleConsensusModelPlane<pcl::PointXYZ>(input_cloud);
            
            pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
            //   selectWithinDistance (const Eigen::VectorXf &model_coefficients, 
            //                const double threshold, 
            //                std::vector<int> &inliers) override;
            //plane_model.selectWithinDistance(groundplane_model_, groundplane_threshold_, inliers->indices);
    float groundplane_threshold_ = 0.3f;
            selectOutsideGroundPlane(input_cloud, groundplane_model_, groundplane_threshold_, inliers->indices);
            
            //pcl::copyPointCloud<pcl::PointXYZ>(*input_cloud, inliers, *output_cloud);
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(input_cloud);
            extract.setIndices(inliers);
            extract.setNegative(true); // points not matching the ground plane
            
            extract.filter (*output_cloud);
            
            return true;
        }
        
void
calculate_obstacle_distances(pcl_ptr pc,
                             std::array<uint16_t, 72>& distances,
                             float fov[2]);

pcl_ptr points_to_pcl(const rs2::points& points)
{
    pcl_ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    auto sp = points.get_profile().as<rs2::video_stream_profile>();
    cloud->width = sp.width();
    cloud->height = sp.height();
    cloud->is_dense = false;
    cloud->points.resize(points.size());
    auto ptr = points.get_vertices();
    for (auto& p : cloud->points)
    {
        p.x = ptr->x;
        p.y = ptr->y;
        p.z = ptr->z;
        ptr++;
    }

    return cloud;
}

int main(int argc, char * argv[]) try
{
    // Declare pointcloud object, for calculating pointclouds and texture mappings
    rs2::pointcloud pc;
    // We want the points object to be persistent so we can display the last cloud when a frame drops
    rs2::points points;

    rs2::decimation_filter dec_filter;
    rs2::temporal_filter temp_filter;
    rs2::hole_filling_filter hole_filter;

    rs2::pipeline pipe;
    rs2::config stream_config;
    std::cout << "Reading from: " << argv[1] << '\n';
    stream_config.enable_device_from_file(argv[1]);
    // stream_config.enable_stream(rs2_stream::RS2_STREAM_DEPTH, 0, 424, 240, rs2_format::RS2_FORMAT_Z16, 30);
    rs2::pipeline_profile selection = pipe.start(stream_config);
    auto depth_stream = selection.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto i = depth_stream.get_intrinsics();
    float fov[2];
    rs2_fov(&i, fov);
    fov[0] = (fov[0] * M_PI)/180.0f;
    fov[1] = (fov[1] * M_PI)/180.0f;
    std::cout << fov[0] << ", " << fov[1] << "\n";

    // Wait for the next set of frames from the camera
    auto frames = pipe.wait_for_frames();

    auto depth = frames.get_depth_frame();
    rs2::frame filtered = depth;

    filtered = dec_filter.process(filtered);
    filtered = temp_filter.process(filtered);
    filtered = hole_filter.process(filtered);

    depth = filtered;

    // Generate the pointcloud and texture mappings
    points = pc.calculate(depth);

    auto pcl_points = points_to_pcl(points);

    pcl_ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::GridMinimum<pcl::PointXYZ> grid_filter(12.0f);
    // grid_filter.setInputCloud(pcl_points);
    // grid_filter.filter(*cloud_filtered);
    // pcl::PassThrough<pcl::PointXYZ> pass;
    // pass.setInputCloud(pcl_points);
    // pass.setFilterFieldName("z");
    // pass.setFilterLimits(0.0, 1.0);
    // pass.filter(*cloud_filtered);

    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(pcl_points);
    voxel_filter.setLeafSize(0.01f,0.01f,0.01f);
    voxel_filter.filter(*cloud_filtered);
    
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    // pcl::IndicesPtr indices (new std::vector <int>);
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        pcl_ptr cloud_p(new pcl::PointCloud<pcl::PointXYZ>);

    int nr_points = (int) cloud_filtered->size();
    // while (cloud_filtered->size() > 0.3*nr_points) {

        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        // seg.setAxis(Eigen::Vector3f(0.0f, 0.0f, 1.0f));
        // seg.setEpsAngle(M_PI/12);
        seg.setDistanceThreshold(0.025);
        seg.setInputCloud(cloud_filtered);
        seg.segment(*inliers, *coefficients);
    
        Eigen::Vector4f ground_coeff(coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);
        remove_groundplane(ground_coeff, cloud_filtered, cloud_p);
          // pcl::ProgressiveMorphologicalFilter<pcl::PointXYZ> pmf;
          // pmf.setInputCloud (cloud_filtered);
          // pmf.setMaxWindowSize (50);
          // pmf.setSlope (1.0f);
          // pmf.setInitialDistance (-1.0f);
          // pmf.setMaxDistance (1.0f);
          // pmf.extract (before_inliers->indices);


        // extract.setInputCloud(cloud_filtered);
        // extract.setIndices(inliers);
        // extract.setNegative(true);
        // extract.filter(*cloud_p);
        // cloud_filtered.swap(cloud_p);
    // }

    // extract.setNegative(false);
    // extract.filter(*cloud_p);
    // cloud_filtered.swap(cloud_p);

    // pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(120.0f);
    // octree.setInputCloud(cloud_p);
    // octree.addPointsFromInputCloud();
    // pcl::PointIndices::Ptr region (new pcl::pointIndices);

    std::array<uint16_t, 72> distances;
    calculate_obstacle_distances(cloud_p, distances, fov);
    // octree.boxSearch

    pcl::visualization::CloudViewer viewer("CloudViewer");
    // viewer.showCloud(pcl_points, "Filtered Cloud");
    // pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = seg.getColoredCloud ();
    viewer.showCloud(cloud_p, "Ground Plane");
    std::cout << "I made it: " << cloud_p->size() << '\n';

    viewer.runOnVisualizationThreadOnce([&fov](pcl::visualization::PCLVisualizer& viewer) {
        Eigen::Affine3f m = Eigen::Affine3f::Identity();
        viewer.removeAllCoordinateSystems();
        viewer.addCoordinateSystem(1.0f, m);
        viewer.setCameraPosition(0, 0, 0, 0, 0, 1, 0, -1, 0);
        viewer.setCameraFieldOfView(1.01256);
        viewer.addCube(-0.166152f, 0.0f,
                       0.0f, 1.28f,
                       0.0f,  4.0f,
                       1.0f,  0.0f, 0.0f);
        viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "cube");
        viewer.setPointCloudRenderingProperties(pcl::visualization::RenderingProperties::PCL_VISUALIZER_COLOR, 1.0f,0.1f,0.3f, "Ground Plane");
    });

    while(!viewer.wasStopped());

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}

void
calculate_obstacle_distances(pcl_ptr pc,
                             std::array<uint16_t, 72>& distances,
                             float fov[2])
{
  Eigen::Affine3f rs_pose =
    static_cast<Eigen::Affine3f>(Eigen::Translation3f(0.0f, 0.0f, 0.0f));
  float angular_res = (float)(1.0f * (M_PI / 180.0f));
  pcl::RangeImage::CoordinateFrame coord_frame = pcl::RangeImage::CAMERA_FRAME;
  float noise_lvl = 0.0f;
  float min_range = 0.0f;
  int border = 0;
  pcl::RangeImage rg_img;
  rg_img.createFromPointCloud(*pc,
                              angular_res,
                              fov[0],
                              fov[1],
                              rs_pose,
                              coord_frame,
                              noise_lvl,
                              min_range,
                              border);
  float* goods = rg_img.getRangesArray();

  float hfov_deg = (fov[0] * 180.0f) / M_PI;
  int rays = distances.size();
  for (int i = 1; i <= rays; i++) {
    int idx = i * (hfov_deg / 72.0f);
    uint16_t depth = UINT16_MAX;
    for (int j = idx; j < i * (hfov_deg / 72.0f); j++) {
      float range = *std::min_element(goods+j*rg_img.height, goods+(j+1)*rg_img.height);
      // rg_img.get1dPointAverage(j, 1, 0, rg_img.height, rg_img.height, ray);
      if (std::isinf(range)) {
        continue;
      }
      else {
        depth = std::min(depth, uint16_t(range*100));
      }
    }
    distances[i - 1] = depth ? depth : 1;
  }
    std::copy(distances.begin(), distances.end(), std::ostream_iterator<float>(std::cout, ","));
    std::cout << '\n';
}
// Registers the state variable and callbacks to allow mouse control of the pointcloud
