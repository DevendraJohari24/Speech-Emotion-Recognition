from flask import Flask, jsonify
import speechEmotion as s
import os
from flask import request

app = Flask(__name__)

@app.route("/")
def hello():
    return os.path.join(os.getcwd(), "temp")

@app.route("/detect", methods=["POST"])
def detect():
    temp_dir = os.path.join(os.getcwd(), "temp")
    if request.method == "POST":
        try:
            save_path = os.path.join(temp_dir, "temp.wav")
            request.files["file"].save(save_path)
            emotion = s.predictEmotion(path=save_path)
            emotionImage = s.getImageUrl(emotion)
            data = {
                "emotion": emotion,
                "emotionImage": emotionImage
            }
            os.remove(save_path)
            return jsonify(data)
        except:
            print("error found")
            data = {
                "emotion": "Not Found",
                "emotionImage": "https://previews.123rf.com/images/kaymosk/kaymosk1804/kaymosk180400006/100130939-error-404-page-not-found-error-with-glitch-effect-on-screen-vector-illustration-for-your-design.jpg"
            }
            return jsonify(data)
        
        
if __name__ == "__main__":
    app.run(debug=False)