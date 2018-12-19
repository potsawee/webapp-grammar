from flask import Flask
app = Flask(__name__)

# use decorators to link the function to a url
@app.route("/")
def hello():
    return "Hello World! - this is WebApp"

if __name__ == '__main__':
    app.run()
