from watson_developer_cloud import SpeechToTextV1
import json, base64
from nltk.tokenize import word_tokenize
import re
import matplotlib.pyplot as plt
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from deepface import DeepFace

#import watson_developer_cloud.natural_language_understanding.features.v1 as \
 #   features
from watson_developer_cloud.natural_language_understanding_v1 import Features, KeywordsOptions, EntitiesOptions, CategoriesOptions, EmotionOptions, SentimentOptions
from wordcloud import WordCloud
import subprocess
##removes non non-alphanumeric characters based on re
def re_nalpha(str):
	pattern = re.compile(r'[^\w\s]', re.U)
	return re.sub(r'_', '', re.sub(pattern, '', str))

def wfreq (text):
	word_dic = {}
	if isinstance(text,str):
		#preprocessing module is used to tokenize the text
		text = word_tokenize(re_nalpha(text))
		for word in text:
			if word not in word_dic:
				word_dic[word] = 1
			else:
				word_dic[word] +=1
	else:
		for sent in text:
			sent_str = word_tokenize(re_nalpha(sent[0]))
			for word in sent_str:
				if word not in word_dic:
					word_dic[word] = 1
				else:
					word_dic[word] +=1
	sorted_list = sorted(word_dic.items(), key=lambda d: d[1],reverse=True)
	return sorted_list

def create_pic(input_text):
	sorted_list=wfreq(input_text)
	print (sorted_list)
	x_list = []
	y_list = []
	for item in sorted_list:
		x_list.append(item[0])
		y_list.append(item[1])
	x=range(len(x_list[:5]))
	plt.bar(x, y_list[:5],width=0.9)
	print(x,x_list)
	plt.xticks((x),x_list)
	plt.savefig('Interviewer/static/Interviewer/userMedia/freq.png')
	plt.close()

def create_emo(input_text):
	natural_language_understanding = NaturalLanguageUnderstandingV1(
		version='2018-08-01',
		url='https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/1e8592f9-5126-42de-aac1-9b9216bac37d',
		iam_apikey="a9dSvu44DdCEYQJmi2Msv5YdAb9jBvovuauLdigl7oP6")

	#print("features",features.Emotion())

	response = natural_language_understanding.analyze(
		text=input_text,
		features=Features(  emotion=EmotionOptions(),))
	result=json.loads(json.dumps(response.result, indent=2))
	print(result)
	emo_rate=result["emotion"]["document"]["emotion"]
	labels = ['sadness', 'joy', 'fear', 'disgust','anger']
	sizes = [emo_rate['sadness'], emo_rate['joy'], emo_rate['fear'], emo_rate['disgust'],emo_rate['anger']]
	colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','black']
	patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
	plt.legend(patches, labels, loc="best")
	plt.tight_layout()
	plt.savefig('Interviewer/static/Interviewer/userMedia/emo.png')
	plt.close()
def word_could(text):
# Generate a word cloud image
	wordcloud = WordCloud().generate(text)
	wordcloud = WordCloud(max_font_size=40).generate(text)
	plt.figure()
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")
	plt.savefig('Interviewer/static/Interviewer/userMedia/wordcloud.png')
	plt.close()

def create_speed(avg_velocity):
	y = []
	y.append(avg_velocity)
	y.append(145)
	x_list = ['Your speed', 'Ideal Speed']
	x = range(len(y))
	plt.bar(x, y, width=0.5)
	plt.xticks((x),x_list)
	plt.savefig('Interviewer/static/Interviewer/userMedia/speed.png')
	plt.close()

def pic_anly():
	result = DeepFace.analyze(img_path = "user_photo.png", actions = ['age', 'gender', 'race', 'emotion'],)
	desc=''    
	if(result):
		desc+="Description for the appearance" + ' : '
		desc+="%s year old %s %s. Emotion:%s"%(result["age"],result["dominant_race"],result["gender"],result["dominant_emotion"])
	else:
		desc="Invalid picture - mutiple faces detected or nofaces detected"
		del(result)
	return (desc)

def speech_to_text(b64code):

	stt = SpeechToTextV1(url='https://api.eu-gb.speech-to-text.watson.cloud.ibm.com/instances/6739751e-9690-493a-84c0-418c1ff62131',
		iam_apikey="tVbCB-1P7r0VMe233CpPeZCInWz1MiHFXXTH1Z5cw8z7")

	audio_file = base64.b64decode(b64code)

	result = json.dumps(stt.recognize(audio_file, content_type="audio/wav",continuous="true",timestamps="true",word_confidence="True",model="en-UK_NarrowbandModel").result, indent=2)
	print(result)
	# print result
	wjdata = json.loads(result)

	temp_str = ''
	speech_speed = ''
	for subs in wjdata["results"]:
		# print "transcript"
		temp_str+=subs["alternatives"][0]["transcript"]
	print(temp_str)
	timestamps = subs["alternatives"][0]["timestamps"]
	average_velocity = round(60/((timestamps[-1][2]-timestamps[0][1])/len(timestamps)))

	if average_velocity<110:
		speech_speed = 'Your speaking rate is slow. You should increase your speaking speed.'
	else:
		if average_velocity>170:
			speech_speed = 'Your speaking rate is fast. You should decrease your speaking speed.'
		else:
			speech_speed = 'You are speaking at the correct rate.'


	print(average_velocity)
	create_pic(temp_str)
	print("temp_str:",temp_str)
	create_emo(temp_str)
	word_could(temp_str)
	create_speed(average_velocity);
	result = {'speech_str': temp_str, 'average_velocity': average_velocity, 'speech_speed': speech_speed }
	return result
