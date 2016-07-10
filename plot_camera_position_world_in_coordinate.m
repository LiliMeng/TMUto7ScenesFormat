close all;
clear;
x = dlmread('world_coordinate_imge_file_x.txt', ' ', 1, 0);
y = dlmread('world_coordinate_imge_file_y.txt', ' ', 1, 0);
z = dlmread('world_coordinate_imge_file_z.txt', ' ', 1, 0);
d = dlmread('depth.txt', ' ', 1, 0);

figure; imagesc(x);  title('X');
figure; imagesc(y);  title('Y');
figure; imagesc(z); colormap('gray'); title('Z');
figure; imagesc(d); colormap('gray'); title('Depth');
