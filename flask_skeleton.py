# import flask dependencies
from flask import Flask, request
import pickle
# initialize the flask app
app = Flask(__name__)
import pandas as pd
df = pd.read_csv("data.csv")
vect1=pickle.load(open("vector.pickel","rb"))
modelKN=pickle.load(open("classifier-kn.pickel","rb"))
vect2=pickle.load(open("vector1.pickel","rb"))

def similarbook(bok):
    y=bok
    if df['title'].eq(bok).any() == False:

        title1=[bok]
        Xtest= vect2.transform(title1)
        y= modelKN.predict(Xtest)
    return y



from sklearn.metrics.pairwise import cosine_similarity
def recommend(title):

    global rec
    # Matching the genre with the dataset and reset the index
    #data = df.loc[df['genre'] == genre]
    #data.reset_index(level = 0, inplace = True)

    # Convert the index into series
    indices = pd.Series(df.index, index = df['title'])

    #Converting the book description into vectors and used bigram
    #tf = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), min_df = 1, stop_words='english')
    #tfidf_matrix = tf.fit_transform(df['cleaned_desc'])
    
    
    # Calculating the similarity measures based on Cosine Similarity
    sg = cosine_similarity(vect1, vect1)

    # Get the index corresponding to original_title
    print(sg.shape)
    idx = indices[title]
    print(idx)
# Get the pairwsie similarity scores
    sig = list(enumerate(sg[idx]))
    print(sig)
# Sort the books
    sig = sorted(sig, key=lambda x: x[1], reverse=True)
    print(sig)
# Scores of the 5 most similar books
    sig = sig[1:6]
# Book indicies
    movie_indices = [i[0] for i in sig]
    print(movie_indices)
    # Top 5 book recommendation
    rec = df[['title']].iloc[movie_indices]
    
    
    # It reads the top 5 recommend book url and print the images
    sentences = []
    indices = 5
    
    begin = 0
    end=0
    for i in range(5):
        end = i + 1
        sentence = ' '.join(list(rec['title'][begin:end]))
        sentences.append(sentence)
        begin = end
    
    x=', '.join(sentences)
    
    return x




# create a route for webhook
@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    fulfillmentText = ''
    query_result = req.get('queryResult')
    if query_result.get('action') == 'get.book':
        ### Perform set of executable code
        ### if required
        ### ...
        #query_result1=query_result.get('action')
        #print(query_result1)
        queryparameter=query_result['parameters']
        #print(queryparameter['address'])
        
        book=similarbook(queryparameter['book'])
        
        if book==queryparameter['book']:
            fulfillmentText = recommend(book)
        else:
            fulfillmentText = recommend(book[0])
        
        #print(fulfillmentText)
    return {
            "fulfillmentText": fulfillmentText,
            "source": "webhookdata"
        }

# run the app
if __name__ == '__main__':
    app.run()
