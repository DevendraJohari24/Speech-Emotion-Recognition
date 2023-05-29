from flask import Flask, jsonify, render_template
import speechEmotion as s
import os
from flask import request

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('pages/index.html')

@app.route("/detect", methods=["POST"])
def detect():
    data = {
        "emotion": 'Not Found',
        "emotionImageUrl": 'https://previews.123rf.com/images/kaymosk/kaymosk1804/kaymosk180400006/100130939-error-404-page-not-found-error-with-glitch-effect-on-screen-vector-illustration-for-your-design.jpg'
    }
    temp_dir = os.path.join(os.getcwd(), "temp")
    if request.method == "POST":
        try:
            save_path = os.path.join(temp_dir, "temp.wav")
            request.files["audio_file"].save(save_path)
            emotion = s.predictEmotion(path=save_path)
            data = {
                'emotion': emotion,
                'emotionImageUrl': "https://cdn.pixabay.com/photo/2017/11/26/15/16/smiley-2979107_1280.jpg"
            }
            os.remove(save_path)
        except:
            print("error found")
            data = {
                "emotion": 'Not Found',
                "emotionImageUrl": 'https://previews.123rf.com/images/kaymosk/kaymosk1804/kaymosk180400006/100130939-error-404-page-not-found-error-with-glitch-effect-on-screen-vector-illustration-for-your-design.jpg'
            }
        return jsonify(data)
    return jsonify(data)
        
if __name__ == "__main__":
    app.run(debug=False)