# USAGE

# python predict_video.py --model model/activity_gpu.model --label-bin model/lb.pickle --input example_clips/shoplifting.mp4 --output output/shoplifting_output.avi --size 10

# import the necessary packages
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
from twilio.rest import Client
camid = 'IRNSS-1I'
location = 'Geosynchronous / 55°E, 29° inclined orbit'

class ModifiedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, groups=None, **kwargs):
        super().__init__(*args, **kwargs)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to  label binarizer")
ap.add_argument("-i", "--input", required=True,
	help="path to our input video")
ap.add_argument("-o", "--output", required=True,
	help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
args = vars(ap.parse_args())

# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
with tf.keras.utils.custom_object_scope({
        'DepthwiseConv2D': ModifiedDepthwiseConv2D
    }):
    # Load the model and labels
        # Try different file extensions
        if args["model"].endswith('.keras'):
            model_path = args["model"]
        elif args["model"].endswith('.h5'):
            model_path = args["model"]
        else:
            # Try adding .keras extension
            model_path = args["model"] + '.keras'
        
        print(f"[INFO] Attempting to load model from: {model_path}")
        model = load_model(model_path, compile=False)
lb = pickle.loads(open(args["label_bin"], "rb").read())

# initialize the image mean for mean subtraction along with the
# predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
client = Client("XXXXXX", "XXXXXX") #update from twilio
prelabel = ''
ok = 'Normal'
fi_label = []
framecount = 0
# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	framecount = framecount+1
	# clone the output frame, then convert it from BGR to RGB
	# ordering, resize the frame to a fixed 224x224, and then
	# perform mean subtraction
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (64, 64)).astype("float32")
	frame -= mean

	# make predictions on the frame and then update the predictions
	# queue
	preds = model.predict(np.expand_dims(frame, axis=0))[0]
#	print('Preds = :', preds)

#	total = (preds[0]+ preds[1]+preds[2] + preds[3]+ preds[4]+preds[5])
#	maximum = max(preds)
#	rest = total - maximum

#	diff = (.8*maximum) - (.1*rest)
#	print('Difference of prob ', diff)
#	th = 100
#	if diff > .60:
#		th = diff
#	print('Old threshold = ', th)


	prediction = preds.argmax(axis=0)
	Q.append(preds)

	# perform prediction averaging over the current history of
	# previous predictions
	results = np.array(Q).mean(axis=0)
	print('Results = ', results)
	maxprob = np.max(results)
	print('Maximun Probability = ', maxprob)
	i = np.argmax(results)
	label = lb.classes_[i]

	rest = 1 - maxprob

	diff = (maxprob) - (rest)
	print('Difference of prob ', diff)
	th = 100
	if diff > .80:
		th = diff


	if (preds[prediction]) < th:
		text = "Alert : {} - {:.2f}%".format((label), 100 - (maxprob * 100))
		cv2.putText(output, text, (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
	else:
		fi_label = np.append(fi_label, label)
		text = "Alert : {} - {:.2f}%".format((label), maxprob * 100)
		cv2.putText(output, text, (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
#		if label != prelabel:
#			client.messages.create(to="<+country code>< receiver mobile number>", #for example +918255555555
#                       from_="+180840084XX", #sender number can be coped from twilo
#                       body='\n'+ str(text) +'\n Satellite: ' + str(camid) + '\n Orbit: ' + location)
		prelabel = label


	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)

	# write the output frame to disk
	writer.write(output)

	# show the output image
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
print('Frame count', framecount)
print('Count label', fi_label)
# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
