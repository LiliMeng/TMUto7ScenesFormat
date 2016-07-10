close all
clear all

%PoseFileName=dir('/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/*.txt');

%numFiles=size(PoseFileName,1);

for i = 1:2486
  poseTQFileName = sprintf('/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/frame-%06d.pose.quaternion.txt', i-1);
  PoseTQ=load(poseTQFileName);
  PoseRT1=PoseTQ2PoseRT(PoseTQ);
  PoseRT=inv(PoseRT1);
  poseRTFileName = sprintf('/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/frame-%06d.pose.txt', i-1);
  fileID = fopen(poseRTFileName,'w');
  for j=1:4
    fprintf(fileID, '%.10f %.10f %.10f %10f \n', PoseRT(j,1), PoseRT(j,2), PoseRT(j,3), PoseRT(j,4));
  end
  fclose(fileID);
end

