import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def obstacle_thresh(img, obstacle_thresh=(0, 160, 0, 160, 0, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    rock_thresh = ~((img[:,:,0] == obstacle_thresh[0]) & (img[:,:,1] == obstacle_thresh[2]) & (img[:,:,2] == obstacle_thresh[4])) \
                            & (img[:,:,0] <= obstacle_thresh[1])\
                            & (img[:,:,1] <= obstacle_thresh[3]) \
                            & (img[:,:,2] <= obstacle_thresh[5])
    # Index the array of zeros with the boolean array and set to 1

    color_select[rock_thresh] = 1
    # Return the binary image
    return color_select
	
def rock_thresh(img, rock_thresh=(140, 180, 130,170,10,30)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    rock_thresh = (img[:,:,0] >= rock_thresh[0]) & (img[:,:,0] < rock_thresh[1])\
                & (img[:,:,1] >= rock_thresh[2]) & (img[:,:,1] < rock_thresh[3]) \
                & (img[:,:,2] >= rock_thresh[4]) & (img[:,:,2] < rock_thresh[5])
    # Index the array of zeros with the boolean array and set to 1
    color_select[rock_thresh] = 1
    # Return the binary image
    return color_select
	
# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # TODO:
    # Convert yaw to radians
    # Apply a rotation
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = xpix*np.cos(yaw_rad)-ypix*np.sin(yaw_rad)
    ypix_rotated = xpix*np.sin(yaw_rad)+ypix*np.cos(yaw_rad)
    # Return the result  
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # TODO:
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):

    if Rover.start_pos == None:
        Rover.start_pos = Rover.pos

    if Rover.perception_count == None:
        Rover.perception_count = 0

    image = Rover.img
    dst_size = 5 
    # Set a bottom offset to account for the fact that the bottom of the image 
    # is not the position of the rover but a bit in front of it
    bottom_offset = 6
    src = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    dst = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
    perspect_img = perspect_transform(Rover.img, src, dst)
	# middle between brightest sky and darkest ground from sample image
    thresh_navi = color_thresh(perspect_img)
    thresh_obst = obstacle_thresh(perspect_img)
    thresh_rock = rock_thresh(perspect_img)

    Rover.vision_image[:,:,0] = thresh_obst*255
    Rover.vision_image[:,:,1] = thresh_rock*255
    Rover.vision_image[:,:,2] = thresh_navi*255

    nav_xpiX, nav_ypiX = rover_coords(Rover.vision_image[:,:,0]) #need to change
    obs_xpiX, obs_ypiX = rover_coords(Rover.vision_image[:,:,1])
    samp_xpiX, samp_ypiX = rover_coords(Rover.vision_image[:,:,2])

    nav_X, nav_Y = pix_to_world(nav_xpiX, nav_ypiX, Rover.pos[0], Rover.pos[1], Rover.yaw, 200, 10)
    obs_X, obs_Y = pix_to_world(obs_xpiX, obs_ypiX, Rover.pos[0], Rover.pos[1], Rover.yaw, 200, 10)
    samp_X, samp_Y = pix_to_world(samp_xpiX, samp_ypiX, Rover.pos[0], Rover.pos[1], Rover.yaw, 200, 10)	

    # Update world map if we are not tilted more than 0.5 deg
    if (Rover.roll < 0.5 or Rover.roll > 359.5) or (Rover.pitch < 0.5 or Rover.pitch > 359.5):

		Rover.worldmap[nav_Y, nav_X, 0] += 1;
        Rover.worldmap[obs_Y, obs_X, 1] += 1;
        Rover.worldmap[samp_Y, samp_X, 2] += 1;
        
    # Clear out low quality nav pixles
    # Delete pixels less than the mean over three
    if(Rover.perception_count % 200 == 0):
        nav_pix = Rover.worldmap[:,:,2] > 0
        lowqual_pix = Rover.worldmap[:,:,2] < np.mean(Rover.worldmap[nav_pix, 2]) / 4
        Rover.worldmap[lowqual_pix, 2] = 0

    dists, angles = to_polar_coords(nav_xpiX, nav_ypiX)

    Rover.nav_dists = dists
    Rover.nav_angles = angles

    samp_dists, samp_angles = to_polar_coords(samp_xpiX, samp_ypiX)

    Rover.samp_dists = samp_dists
    Rover.samp_angles = samp_angles
    
    Rover.perception_count += 1

    return Rover

