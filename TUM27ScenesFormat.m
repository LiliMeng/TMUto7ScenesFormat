close all
clear all
PoseTQ=load('/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/frame-000000.pose.quaternion.txt')

PoseRT=PoseTQ2PoseRT(PoseTQ)

fileID=fopen('/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/frame-000000.pose.txt','w');

for i=1:4
    fprintf(fileID, '%.10f %.10f %.10f %10f \n', PoseRT(i,1), PoseRT(i,2), PoseRT(i,3), PoseRT(i,4));
end
