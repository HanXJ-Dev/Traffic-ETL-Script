
import cv2
import json
import pandas
import cvlib as cv
import requests
import os
from cvlib.object_detection import draw_bbox


'''
	Getting response from API
'''
response_obj = requests.get(
	'https://api.data.gov.sg/v1/transport/traffic-images')

response_json = response_obj.json()

'''
	Getting the timestamp of the data received to segregate data information
'''
timestamp = response_json['items'][0]['timestamp']
os.mkdir(timestamp)

'''
	Writing the response as it is
'''
with open(os.path.join(timestamp, 'response.json'), 'w') as w:
	w.write(json.dumps(response_json))
	w.close()

'''
	Retrieving the image from the image source
	provided in the response
'''
image_folder = os.path.join(timestamp, 'images')
os.mkdir(image_folder)

'''
	Creating data frame to store transformed data
'''
pd_columns=['camera_id', 'image', 'latitude', 'longtitude', 'number_of_vehicles', 'timestamp']
pd_data = []

'''
	Looping through all the cameras from the response
'''
for camera in response_json['items'][0]['cameras']:
	'''
		Writing the image from image URL
	'''
	file_name = f"{os.path.join(image_folder, camera['camera_id'])}"
	with open(f'{file_name}.jpeg', 'wb') as handle:
		response = requests.get(camera['image'], stream=True)
		if not response.ok:
			break
		for block in response.iter_content(1024):
			if not block:
				break
			handle.write(block)
			
		'''
			Using cv2 to detect and count the number of vehicles
			based on number of: Cars + Truck + Motorcycle
		'''
		im = cv2.imread(f'{file_name}.jpeg')
		bbox, label, conf = cv.detect_common_objects(im)
		output_image = draw_bbox(im, bbox, label, conf)
		number_of_vehicles = str(label.count('car')+label.count('truck')+label.count('motorcycle'))

		'''
			Writing seperate image that is labelled
		'''
		cv2.imwrite(f'{file_name}_labelled.jpeg', output_image)


		'''
			Writing the data frame 
		'''
		pd_data.append([
			camera['camera_id'], 
			f'{file_name}.jpeg', 
			camera['location']['latitude'], 
			camera['location']['longitude'],
			number_of_vehicles,
			camera['timestamp']])


'''
	Write into a csv file
'''
df = pandas.DataFrame(pd_data, columns=pd_columns)
df.to_csv(f'{timestamp}/data.csv', index=False)
