import yolov5

# load model
model = yolov5.load('best.pt')
  
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image




import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Load the point cloud
pcd = o3d.io.read_point_cloud('rgbd.ply')



# Extract the XYZ coordinates
xyz = np.asarray(pcd.points)

# For a simple orthographic projection, we can just discard the Z component
xy = xyz[:, :2]

# We can also extract the RGB colors
colors = np.asarray(pcd.colors)

# Now we can create a scatter plot, using the XY coordinates for position and the RGB colors
plt.scatter(xy[:, 0], xy[:, 1], c=colors)
plt.savefig('fig1.png', dpi=300)



# Extract the RGB data
colors = np.asarray(pcd.colors)

print(colors.shape)

# Reshape and normalize the data to be in range [0, 255]
rgb_image = colors.reshape((424, 240, 3)) * 255



# set image
img = rgb_image

# perform inference
results = model(img)

# inference with larger input size
results = model(img, size=640)

# inference with test time augmentation
results = model(img, augment=True)

# parse results
predictions = results.pred[0]
boxes = predictions[:, :4] # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

# show detection bounding boxes on image
results.show()

# save results into "results/" folder
results.save(save_dir='results/')
