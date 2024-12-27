from flask import Flask,render_template,request

##Jinja 2 Template Engine
'''
{{ }} expressions to print output in html
{%...%} conditions, for loops
{#...#} this is for comments
'''

### WSGI Application
app=Flask(__name__)

@app.route("/")
def welcome():
    return "Weclome to this Course! This is an awsome course"

@app.route("/index",methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/submit",methods=['GET','POST'])
def submit():
    if request.method=='POST':
        name=request.form['name']
        return f'Hello, {name} to my world!'
    return render_template("form.html")

# Variable rule
@app.route('/sucess/<int:score>')
def sucess(score):
    res=""
    if score>=50:
        res="PASS"
    else:
        res="FAIL"

    return render_template("result.html",results=res)

@app.route('/sucess1/<int:score>')
def sucess1(score):
    res=""
    if score>=50:
        res="PASS"
    else:
        res="FAIL"
    
    exp = {'score':score,'res':res}

    return render_template("result1.html",results=exp)

if __name__=='__main__':
    app.run(debug=True)