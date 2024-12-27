from flask import Flask

### WSGI Application
app=Flask(__name__)

@app.route("/")
def welcome():
    return "Weclome to this Course! This is an awsome course"


if __name__=='__main__':
    app.run(debug=True)