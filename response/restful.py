from flask import Flask,request
#import sqlalchemy
from json import dumps
import jsonify
from flask_restful import Resource, Api
from detection.image import recognize

app = Flask(__name__)
api = Api(app)

class HumansCount(Resource):
	def get(self):
		'''
		Test GET
		'''
		return {"hello": "oke"}
	def post(self,img_path):
		"""
		receive a post request and respones a dictionary with 2 key
		@count: number of peoples in images
		@path: path of images to show

		Input Param:
		@params --img_path, input image path,

		"""
		count, img = recognize(img_path)
		new_path = img_path + img_path.split('.')[0].split('input\\')[-1]
		cv2.imwrite('output\\{new_path}.jpg')
		dic = {'count': count, 'path': 'output\\{new_path}.jpg'}
		#do tao khong tra ve anh duoc nen tao tra ve relative path cua cai anh, m dung no de lay anh roi process len nha :v
		return dic


api.add_resource(HumansCount, '/humancounting') # Human counting route


if __name__=='__main__':
	app.run(debug=True)
