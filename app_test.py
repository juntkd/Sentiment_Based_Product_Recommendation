app = Flask(__name__)


@app.route('/')
def home():

    print('hellow world')


if __name__ == '__main__':
    app.run(debug=True)