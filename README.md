# Deep SEO

- Description: Final project for Le Wagon Tokyo Bootcamp, for this project we developed 2 deep learning models using Tensorflow Keras to analize the listing of a product in Amazon USA marketplace to evaluate the quality of the SEO (Search Engine Optimization) for a new product in the category of Cellphones and Accessories, taking as features in the first place Text that is analized with a RNN model and images throught a CNN model, in both cases the output we look to predict is the product selling ranking that was divided in 10 categories wich both models classify the Text features and Image features. 

- Backend : 
  - Tensorflow Keras
  - Fast API
  - Uvicorn
  - Docker
  - GCP
- Frontend:
  - Streamlit
  - Heroku 
- Data Source: https://nijianmo.github.io/amazon/index.html
- Authors:
  - <a href="https://github.com/Elizabeth-kok">Elizabeth Kok</a>
  - <a href="https://github.com/na0young124">Nayoung Kim</a>
  - <a href="https://github.com/Cris-CL">Cristobal Cepeda</a>

<a href="https://deep-seo-app.herokuapp.com/">Deep SEO Heroku app</a>

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for deep_seo in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/deep_seo`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "deep_seo"
git remote add origin git@github.com:{group}/deep_seo.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
deep_seo-run
```

# Install

Go to `https://github.com/{group}/deep_seo` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/deep_seo.git
cd deep_seo
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
deep_seo-run
```
