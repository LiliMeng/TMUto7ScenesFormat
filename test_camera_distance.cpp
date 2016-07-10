//
//  test_camera_distance.cpp
//  RGBRegressionForest
//
//  Created by Lili on 7/3/16.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "test_camera_distance.h"


struct mycomparision1 {
    bool operator() (double i, double j) {return (i>j);}
}mycompareLS;

struct mycomparision2 {
    bool operator() (double i,double j) { return (i<j);}
} mycompareSL;

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {
    
    //initialize original index locations
    vector<size_t> index(v.size());
    for(size_t i=0; i !=index.size(); ++i)
    {
        index[i]=i;
    }
    
    //sort indexes based on comparing values in v
    sort(index.begin(), index.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; } );
    
    return index;
    
}



void camera_rot_distance(const char* train_file_name,
                         const char* test_file_name,
                         const double trans_threshold,
                         vector<double> &rot_vec_within_trans_threshold)
{
    vector<string> train_pose_files = Ms7ScenesUtil::read_file_names(train_file_name);
    vector<string> test_pose_files  = Ms7ScenesUtil::read_file_names(test_file_name);
    
    
    vector<cv::Mat> train_poses_vec, test_poses_vec;
    for(int i=0; i<test_pose_files.size(); i++)
    {
        const char* test_pose_file = test_pose_files[i].c_str();
        
        cv::Mat test_camera_pose=Ms7ScenesUtil::read_pose_7_scenes(test_pose_file);
        
        test_poses_vec.push_back(test_camera_pose);
        
    }
    
    for(int i=0; i<train_pose_files.size(); i++)
    {
        const char* train_pose_file=train_pose_files[i].c_str();
        
        cv::Mat train_camera_pose=Ms7ScenesUtil::read_pose_7_scenes(train_pose_file);
        
        train_poses_vec.push_back(train_camera_pose);
        
    }
    
    
    for(int k=0; k<test_poses_vec.size(); k++)
    {
        cv::Mat pose1=test_poses_vec[k];
        //cout<<"the pose file from the test sequence is "<<test_pose_files[k]<<endl;
        
        for(int i=0; i<train_poses_vec.size();i++)
        {
            
            double rot_error = 0.0;
            double trans_error = 0.0;
            
            
            CvxPoseEstimation::poseDistance(pose1,
                                            train_poses_vec[i],
                                            rot_error,
                                            trans_error);
            
        
            if (trans_error < trans_threshold) {
                
                rot_vec_within_trans_threshold.push_back(rot_error);
            }
        }
        
        
        std::sort(rot_vec_within_trans_threshold.begin(), rot_vec_within_trans_threshold.end(), mycompareSL);
        
        
        if(rot_vec_within_trans_threshold.size()>=1)
        {
            
            printf("within the trans threshold of %f, the minimum rot error is %lf\n", trans_threshold, rot_vec_within_trans_threshold[0]);
        
        }
        else
        {
            printf("please increase the trans threshold");
        }
    
    }
}


void camera_trans_distance(const char* train_file_name,
                           const char* test_file_name,
                           const double rot_threshold,
                           vector<double> &test_min_trans_vec_within_rot_threshold)
{

    vector<string> train_pose_files = Ms7ScenesUtil::read_file_names(train_file_name);
    vector<string> test_pose_files = Ms7ScenesUtil::read_file_names(test_file_name);

    
    vector<cv::Mat> train_poses_vec, test_poses_vec;
    for(int i=0; i<test_pose_files.size(); i++)
    {
        const char* test_pose_file = test_pose_files[i].c_str();
        
        cv::Mat test_camera_pose=Ms7ScenesUtil::read_pose_7_scenes(test_pose_file);
        
        test_poses_vec.push_back(test_camera_pose);
        
    }
    
    for(int i=0; i<train_pose_files.size(); i++)
    {
        const char* train_pose_file=train_pose_files[i].c_str();
        
        cv::Mat train_camera_pose=Ms7ScenesUtil::read_pose_7_scenes(train_pose_file);
        
        train_poses_vec.push_back(train_camera_pose);
       
    }

    vector<double> inlier_rot_test, inlier_trans_test;
    
    //suppose we just pick up the pose1 from the test files as our query pose, and see the range of other poses in the training set compared with pose1
    
    for(int k=0; k<test_poses_vec.size(); k++)
    {
        cv::Mat pose1=test_poses_vec[k];
        //cout<<"the pose file from the test sequence is "<<test_pose_files[k]<<endl;
    
        vector<double> rot_error_vec, trans_error_vec, trans_error_within_rot_threshold;
    
        vector<size_t> trans_error_rot_threshold_indexes;
        for(int i=0; i<train_poses_vec.size();i++)
        {
        
            double rot_error = 0.0;
            double trans_error = 0.0;
            
            
            CvxPoseEstimation::poseDistance(pose1,
                                            train_poses_vec[i],
                                            rot_error,
                                            trans_error);
        
            rot_error_vec.push_back(rot_error);
            trans_error_vec.push_back(trans_error);
            if (rot_error < rot_threshold) {
                trans_error_rot_threshold_indexes.push_back(i);
                trans_error_within_rot_threshold.push_back(trans_error);
            }
        }
    
        
        std::sort(trans_error_within_rot_threshold.begin(), trans_error_within_rot_threshold.end(), mycompareSL);
        
        vector<size_t> sorted_index_vec=sort_indexes(trans_error_within_rot_threshold);
        
        if(trans_error_within_rot_threshold.size()>=1)
        {
    
            //printf("within the rot threshold of %f, the minimum trans error is %lf\n", rot_threshold, trans_error_within_rot_threshold[0]);
            
           // test_min_trans_vec_within_rot_threshold.push_back(trans_error_within_rot_threshold[0]);
            test_min_trans_vec_within_rot_threshold.push_back(trans_error_within_rot_threshold[0]);
            
            if(trans_error_within_rot_threshold.size()==1)
            {
                printf("within the rot threshold of %f, the minimum trans error is %lf\n", rot_threshold, trans_error_within_rot_threshold[sorted_index_vec[0]]);
                cout<<"the pose file is "<<train_pose_files[trans_error_rot_threshold_indexes[sorted_index_vec[0]]]<<endl;
    
            }
            else if(trans_error_within_rot_threshold.size()==2)
            {
            
                printf("within the rot threshold of %f, the minimum trans error is %lf\n", rot_threshold, trans_error_within_rot_threshold[sorted_index_vec[0]]);
                cout<<"the pose file is "<<train_pose_files[trans_error_rot_threshold_indexes[sorted_index_vec[0]]]<<endl;
                printf("within the rot threshold of %f, the second minimum trans error is %lf\n", rot_threshold, trans_error_within_rot_threshold[sorted_index_vec[1]]);
                cout<<"the pose file is "<<train_pose_files[trans_error_rot_threshold_indexes[sorted_index_vec[1]]]<<endl;
            }
            else if(trans_error_within_rot_threshold.size()>=3)
            {
                printf("within the rot threshold of %f, the minimum trans error is %lf\n", rot_threshold, trans_error_within_rot_threshold[sorted_index_vec[0]]);
                cout<<"the pose file is "<<train_pose_files[trans_error_rot_threshold_indexes[sorted_index_vec[0]]]<<endl;
                printf("within the rot threshold of %f, the 2nd minimum trans error is %lf\n", rot_threshold, trans_error_within_rot_threshold[sorted_index_vec[1]]);
                cout<<"the pose file is "<<train_pose_files[trans_error_rot_threshold_indexes[sorted_index_vec[1]]]<<endl;
                printf("within the rot threshold of %f, the 3rd minimum trans error is %lf\n", rot_threshold, trans_error_within_rot_threshold[sorted_index_vec[2]]);
                cout<<"the pose file is "<<train_pose_files[trans_error_rot_threshold_indexes[sorted_index_vec[2]]]<<endl;
            }
       }
    }
}

void test_min_camera_rotDistanceUnderTransThreshold()
{
    ofstream fout1("/Users/jimmy/Desktop/RGBTrainChess/rotDiswithTransError20_fire.txt");
    ofstream fout2("/Users/jimmy/Desktop/RGBTrainChess/rotDiswithTransError30_fire.txt");
    ofstream fout3("/Users/jimmy/Desktop/RGBTrainChess/rotDiswithTransError50_fire.txt");
    
    // read ground truth data for chess
    //const char* train_file_name="/Users/jimmy/Desktop/RGBTrainChess/train_4000_chess/camera_pose_list.txt";
    //const char* test_file_name = "/Users/jimmy/Desktop/RGBTrainChess/test_2000_chess/camera_pose_list.txt";
    
    ///for heads
    //const char* train_file_name="/Users/jimmy/Desktop/RGBTrainHeads/train_1000/camera_pose_list.txt";
    //const char* test_file_name = "/Users/jimmy/Desktop/RGBTrainHeads/test_1000/camera_pose_list.txt";
    
    ///for fire
    const char* train_file_name="/Users/jimmy/Desktop/RGBTrainFire/camera_pose_list.txt";
    const char* test_file_name = "/Users/jimmy/Desktop/RGBTrainFire/test/camera_pose_list.txt";
    
    double trans_threshold1=0.20;
    double trans_threshold2=0.30;
    double trans_threshold3=0.50;
    
    vector<double> rot_vec_within_trans_threshold1, rot_vec_within_trans_threshold2, rot_vec_within_trans_threshold3;
    camera_rot_distance(train_file_name,
                          test_file_name,
                          trans_threshold1,
                          rot_vec_within_trans_threshold1);
    
    camera_rot_distance(train_file_name,
                          test_file_name,
                          trans_threshold2,
                          rot_vec_within_trans_threshold2);
    
    camera_rot_distance(train_file_name,
                          test_file_name,
                          trans_threshold3,
                          rot_vec_within_trans_threshold3);
    
    
    for(int i=0; i<rot_vec_within_trans_threshold1.size();i++)
    {
        
        fout1<<rot_vec_within_trans_threshold1[i]<<endl;
    }
    
    
    for(int i=0; i<rot_vec_within_trans_threshold2.size();i++)
    {
        
        fout2<<rot_vec_within_trans_threshold2[i]<<endl;
    }
    
    
    for(int i=0; i<rot_vec_within_trans_threshold3.size();i++)
    {
        fout3<<rot_vec_within_trans_threshold3[i]<<endl;
    }
    

}


void test_camera_distance()
{
    ofstream fout1("/Users/jimmy/Desktop/RGBTrainChess/test_trans_error5_fire.txt");
    ofstream fout2("/Users/jimmy/Desktop/RGBTrainChess/test_trans_error10_fire.txt");
    ofstream fout3("/Users/jimmy/Desktop/RGBTrainChess/test_trans_error15_fire.txt");
    
    // read ground truth data for chess
    //const char* train_file_name="/Users/jimmy/Desktop/RGBTrainChess/train_4000_chess/camera_pose_list.txt";
    //const char* test_file_name = "/Users/jimmy/Desktop/RGBTrainChess/test_2000_chess/camera_pose_list.txt";
  
    ///for heads
    //const char* train_file_name="/Users/jimmy/Desktop/RGBTrainHeads/train_1000/camera_pose_list.txt";
    //const char* test_file_name = "/Users/jimmy/Desktop/RGBTrainHeads/test_1000/camera_pose_list.txt";
    
    ///for fire
    const char* train_file_name="/Users/jimmy/Desktop/RGBTrainFire/camera_pose_list.txt";
    const char* test_file_name = "/Users/jimmy/Desktop/RGBTrainFire/test/camera_pose_list.txt";
   
    double rot_threshold1=5;
    double rot_threshold2=10;
    double rot_threshold3=15;
    
    vector<double> test_min_trans_vec_within_rot_threshold1, test_min_trans_vec_within_rot_threshold2, test_min_trans_vec_within_rot_threshold3;
    camera_trans_distance(train_file_name,
                          test_file_name,
                          rot_threshold1,
                          test_min_trans_vec_within_rot_threshold1);
    
    camera_trans_distance(train_file_name,
                          test_file_name,
                          rot_threshold2,
                          test_min_trans_vec_within_rot_threshold2);
    
    camera_trans_distance(train_file_name,
                          test_file_name,
                          rot_threshold3,
                          test_min_trans_vec_within_rot_threshold3);
    
    
    for(int i=0; i<test_min_trans_vec_within_rot_threshold1.size();i++)
    {

        fout1<<test_min_trans_vec_within_rot_threshold1[i]<<endl;
    }
    
    
    for(int i=0; i<test_min_trans_vec_within_rot_threshold2.size();i++)
    {
        
        fout2<<test_min_trans_vec_within_rot_threshold2[i]<<endl;
    }
    
    
    for(int i=0; i<test_min_trans_vec_within_rot_threshold3.size();i++)
    {
        fout3<<test_min_trans_vec_within_rot_threshold3[i]<<endl;
    }
    
}


void test_minCameraDistanceUnderAngularThreshold()
{
    ///for chess
    const char* train_file_name="/Users/jimmy/Desktop/RGBTrainChess/train_4000_chess/camera_pose_list.txt";
    const char* test_file_name = "/Users/jimmy/Desktop/RGBTrainChess/test_2000_chess/camera_pose_list.txt";
    
    
    
    double angular_threshold = 0.3;
    vector<string> train_pose_files =Ms7ScenesUtil::read_file_names(train_file_name);
    vector<string> test_pose_files = Ms7ScenesUtil::read_file_names(test_file_name);
    
    vector<cv::Mat> train_poses_vec, test_poses_vec;
    
    for(int i=0; i<train_pose_files.size(); i++)
    {
        const char* train_pose_file=train_pose_files[i].c_str();
        
        cv::Mat train_camera_pose=Ms7ScenesUtil::read_pose_7_scenes(train_pose_file);
        
        train_poses_vec.push_back(train_camera_pose);
        
    }
    
    for(int i=0; i<test_pose_files.size(); i++)
    {
        const char* test_pose_file = test_pose_files[i].c_str();
        
        cv::Mat test_camera_pose=Ms7ScenesUtil::read_pose_7_scenes(test_pose_file);
        
        //test_poses_vec.push_back(test_camera_pose);
        double minCameraDistance=CvxPoseEstimation::minCameraAngleUnderTranslationalThreshold(train_poses_vec, test_camera_pose, angular_threshold);
        printf("Under angular threshold %lf, min camera distance is: %lf\n", angular_threshold, minCameraDistance);
    }
}

void test_cameraDepthToWorldCoordinate()
{
    Mat calibration_matrix = cv::Mat::eye(3, 3, CV_64F);
    calibration_matrix.at<double>(0, 0) = 535.4;
    calibration_matrix.at<double>(1, 1) = 539.2;
    calibration_matrix.at<double>(0, 2) = 320.1;
    calibration_matrix.at<double>(1, 2) = 247.6;
    
    const char* depth_file_name="/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/frame-000000.depth.png";
    const char* pose_file_name="/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/frame-000000.pose.txt";
    
    cv::Mat camera_depth_img;
    CvxIO::imread_depth_16bit_to_64f(depth_file_name, camera_depth_img);
    cv::Mat camera_to_world_pose = Ms7ScenesUtil::read_pose_7_scenes(pose_file_name);
    double depth_factor = 5000;
    double min_depth = 0.05;
    double max_depth = 10.0;
    
    cv::Mat camera_coordinate;
    cv::Mat mask;
    
    cv::Mat world_coordinate_image=CvxCalib3D::cameraDepthToWorldCoordinate(camera_depth_img,
                                                                            camera_to_world_pose,
                                                                            calibration_matrix,
                                                                            depth_factor,
                                                                            min_depth,
                                                                            max_depth,
                                                                            camera_coordinate,
                                                                            mask);
    
  
    vector<Mat> world_coordinate_img_split;
    cv::split(world_coordinate_image,world_coordinate_img_split);
    cv::Mat world_coordinate_image_x=world_coordinate_img_split[0];
    cv::Mat world_coordinate_image_y=world_coordinate_img_split[1];
    cv::Mat world_coordinate_image_z=world_coordinate_img_split[2];
   
    
    
    const char* world_coordinate_imge_file_x="/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/world_coordinate_imge_file_x.txt";
    const char* world_coordinate_imge_file_y="/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/world_coordinate_imge_file_y.txt";
    const char* world_coordinate_imge_file_z="/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/world_coordinate_imge_file_z.txt";
    const char* world_coordinate_imge_file_depth="/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/depth.txt";
    
    
    CvxIO::save_mat(world_coordinate_imge_file_x, world_coordinate_image_x);
    CvxIO::save_mat(world_coordinate_imge_file_y, world_coordinate_image_y);
    CvxIO::save_mat(world_coordinate_imge_file_z, world_coordinate_image_z);
    CvxIO::save_mat(world_coordinate_imge_file_depth, camera_depth_img);

}

