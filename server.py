from flask import Flask, render_template,Response
import os

app=Flask(__name__)


@app.route('/')
def index():
    #rendering webpage
    return render_template('index.html')

@app.route('/Emojify', methods=['POST'])
def Emojify():
    os.system('python WithoutGUI.py')
    return render_template('Emojify.html')

if __name__=='__main__':
    #defining server ip address and port
    app.run(port=4555,debug=True)