#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore') # Turn off warnings
plt.style.use('seaborn-white') # Use seaborn-style plots
#for interactive plots
import ipywidgets
from ipywidgets import interact
from ipywidgets import interact_manual
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#importing the dataset
books = pd.read_csv('Books.csv')
ratings = pd.read_csv('Ratings.csv')
users = pd.read_csv('Users.csv')


# In[3]:


books.head()


# In[4]:


ratings.head()


# In[5]:


users.head()


# In[6]:


books.shape


# In[7]:


ratings.shape


# In[8]:


users.shape


# In[9]:


#analysing the users dataset
users.columns = users.columns.str.strip().str.lower().str.replace('-', '_')
users.head()


# In[10]:


users.dtypes


# In[11]:


uniq_users = users.user_id.nunique()
all_users = users.user_id.count()
print(f'No. of unique user_id entries: {uniq_users} | Total user_id entries: {all_users}')


# ## #The 'User-ID' field is integers, 'Location' are strings', and the 'Age' values are floats.
# 
# ### #We can probably change the ages to ints. Let's take a look at the values first.

# In[12]:


print(sorted(users.age.unique()))


# The Age range goes from 0 to 244 years old! obviously this can't be correct so we will set all the ages less than 5 and greater than 100 to NaN
# 

# In[13]:


users.loc[(users.age<5) | (users.age>100), 'age'] = np.nan


# In[14]:


users.head()


# In[15]:


print(sorted(users.age.unique()))


# As the ages now go from 5 to 100 ,with missing values as NaN let us plot this to see how they are distributed

# In[16]:


#Creating the Histogram for the Age Field
hist = users.age.hist(bins=10, figsize=(12,5))
hist.set_xlabel('Age')
hist.set_ylabel('counts')
hist.set_xticks(range(0,110,10))
plt.show()


#  It seems that most of our reviewers are in their late 20s to early 30s.
# 
# How many missing Age's do we now have in the dataset?

# In[17]:


age_null = users.age.isnull().sum() # Sums up the 1's returned by the isnull() mask
all_users = users.user_id.count() # Counts the number of cells in the series - excludes NaNs!
print(f'There are {age_null} empty age values in our set of {all_users} users (or {(age_null/all_users)*100:.2f}%).')


# Next, can we expand the 'Location' field to break it up into 'City', 'State', and 'Country'.

# In[18]:


# Note: Used Pandas Series.str.split method as it has an 'expand' parameter which can handle None cases
user_location_expanded = users.location.str.split(',', n=2, expand=True)
user_location_expanded.columns = ['city', 'state', 'country']
users = users.join(user_location_expanded)


# In[19]:


users


# In[20]:


#Let's take a quick look at these Location-derived fields.

top_cities = users.city.value_counts().head(10)
print(f'The 10 cities with the most users are:\n{top_cities}')


# In[21]:


top_countries = users.country.value_counts().head(10)
print(f'The 10 countries with the most users are:\n{top_countries}')


# It looks like an empty 'Country' field is in the top 10 most common entries

# In[22]:


empty_string_country = users[users.country == ''].country.count()
nan_country = users.country.isnull().sum()
print(f'There are {empty_string_country} entries with empty strings, and {nan_country} NaN entries in the Country field')


# In[23]:


#We should probably change these empty strings to NaNs.

users.country.replace('', np.nan, inplace=True)

# Check for invalid entries in the 'location' column
invalid_entries = users['country'].isin(['', 'Unknown', 'N/A', 'NaN', 'None'])



# In[24]:


invalid_entries.sum()


# In[25]:


# Calculate the percentage of invalid entries
percentage_invalid_entries = invalid_entries.sum() / users['country'].count() * 100

print(f'Percentage of invalid entries in the "country" column: {percentage_invalid_entries}')


# Might want to filter out invalid Location-based entries by looking at the city/state/country entries that only occur a few times (maybe <3), however we will leave this as is at the moment. I'll re-visit this idea more thoroughly if we end up using Location in the modeling

# In[26]:


users.head(10)


# In[27]:


#Bar Plot of Top 10 Most Active User countries
top_user_countries = users['country'].value_counts().head(10)
plt.bar(top_user_countries.index,top_user_countries.values)
plt.xticks(rotation = 90)
plt.xlabel('country')
plt.ylabel('Count')
plt.title('Top 10 Most Active User Countries')
plt.show()


# Verifying and Analysing the Books DataFrame

# In[28]:


books.head()


# In[29]:


#note that ISBN column may contain some letters so we can not convert them into numeric values
#we probably want year-of-pulication to be in ints/floats and rest to be in objects datatype


# In[30]:


books.dtypes


# In[31]:


books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')


# In[32]:


# Check for 0's or NaNs in Year of Publication
zero_yr = books[books['Year-Of-Publication'] == 0]['Year-Of-Publication'].count()
nan_yr = books['Year-Of-Publication'].isnull().sum()
print(f'There are {zero_yr} entries as \'0\', and {nan_yr} NaN entries in the Year of Publication field')


# In[33]:


# Replace all years of zero with NaN
books['Year-Of-Publication'].replace(0, np.nan, inplace=True)


# In[34]:


#line plot of Book Publication Trends Over Years
books.groupby('Year-Of-Publication')['Book-Title'].agg('count').plot(figsize = (20,10))
plt.xlabel('Year-Of-Publication')
plt.ylabel('Count')
plt.title('Book Publication Trends Over Years')
plt.show()


# In[35]:


#We clean up the ampersand formatting in the Publisher field.

books.Publisher = books.Publisher.str.replace('&amp', '&', regex=False)
books.head()


# In[36]:


#Check that there are no duplicated book entries.
uniq_books = books.ISBN.nunique()
all_books = books.ISBN.count()
print(f'No. of unique books: {uniq_books} | All book entries: {all_books}')


# In[37]:


#Let's look at the most frequent Publishing houses in the dataset.

top_publishers = books.Publisher.value_counts()[:10]
print(f'The 10 publishers with the most entries in the books table are:\n{top_publishers}')


# In[38]:


top_authors = books['Book-Author'].value_counts()[:10]
print(f'The 10 authors with the most entries in the books table are:\n{top_authors}')


# In[39]:


#We should search for empty or NaN values in these fields too.
empty_string_publisher = books[books.Publisher == ''].Publisher.count()
nan_publisher = books.Publisher.isnull().sum()
print(f'There are {empty_string_publisher} entries with empty strings, and {nan_publisher} NaN entries in the Publisher field')


# There are 0 entries with empty strings, and 2 NaN entries in the Publisher field
#  no empty strings in the Publisher field, and only 2 NaNs.

# In[40]:


empty_string_author = books[books['Book-Author'] == '']['Book-Author'].count()
nan_author = books['Book-Author'].isnull().sum()
print(f'There are {empty_string_author} entries with empty strings, and {nan_author} NaN entries in the Author field')


# In[41]:


# Filter the data to exclude empty strings and NaN entries in the 'Book-Author' field
filtered_authors = books[(books['Book-Author'] != '') & (~books['Book-Author'].isnull())]

# Find the top 10 authors
top_10_authors = filtered_authors['Book-Author'].value_counts().head(10)
print(top_10_authors)


# In[42]:


top_10_authors = books['Book-Author'].value_counts().head(10)
plt.bar(top_10_authors.index, top_10_authors.values)
plt.xticks(rotation=90)
plt.xlabel('Author')
plt.ylabel('Count')
plt.title('Top 10 Authors')
plt.show()


# In[43]:


#Let's look at the titles.

top_titles = books['Book-Title'].value_counts().head(10)
print(f'The 10 book titles with the most entries in the books table are:\n{top_titles}')


# This is actually quite an important observation. Although all of the ISBN entries are unique in the 'books' dataframe, different forms of the same book will have different ISBNs - i.e. paperback, e-book, etc. Therefore, we can tell and see that some books have multiple ISBN entries.
# 

# In[44]:


top_10_titles = books['Book-Title'].value_counts().head(10)
plt.bar(top_10_titles.index, top_10_titles.values)
plt.xticks(rotation=90)
plt.xlabel('Book Title')
plt.ylabel('Count')
plt.title('Top 10 Titles')
plt.show()


# In[45]:


books[books['Book-Title']=='Pride and Prejudice']


# It looks like each ISBN assigned to the book 'Pride and Prejudice' has different Publisher and Year of Publication values also.
# 
# It might be more useful for our model if we simplified this to give each book a unique identifier, independent of the book format, as our recommendations will be for a book, not a specific version of a book.

# In[46]:


#Verifying and Analysing Ratings Table

ratings.columns = ratings.columns.str.strip().str.lower().str.replace("-","_")
ratings.head()


# In[47]:


ratings.dtypes


# In[48]:


# Which users contribute the most ratings?

Top_users = ratings.groupby('user_id').isbn.count().sort_values(ascending=False)
print(f'The 20 users with the most ratings:\n{Top_users[:20]}')


# In[49]:


# Let's see how they are distributed.

# user distribution - users with more than 50 ratings removed
user_hist = Top_users.where(Top_users<50)
user_hist.hist(bins=30)
plt.xlabel('No. of ratings')
plt.ylabel('count')
plt.show()


# It looks like by far the most frequent events are users with only 1 or 2 rating entries. We can see that the 'Top users' with thousands of ratings are significant outliers.
# 
# This becomes clear if we make the same histogram with a cutoff for users with a minimum of 1000 ratings.

# In[50]:


# only users with more than 1000 ratings
Top_user_hist = Top_users.where(Top_users>1000)
Top_user_hist.hist(bins=30)
plt.xlabel('No. of ratings (min. 1000)')
plt.ylabel('count')
plt.show()


# In[51]:


# Let's see what the distribution of ratings looks like.
Rating = ratings['book_rating'].value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.rcParams.update({'font.size': 15}) # Set larger plot font size
plt.bar(Rating.index, Rating.values)
plt.xlabel('Rating')
plt.ylabel('counts')
plt.show()


# In[52]:


print(f'Size of book_ratings before removing zero ratings: {len(ratings)}')


# In[53]:


book_ratings = ratings[ratings['book_rating'] != 0]
print(f'Size of book_ratings after removing zero ratings: {len(book_ratings)}')


# In[54]:


rtg = book_ratings['book_rating'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
plt.rcParams.update({'font.size': 15}) # Set larger plot font size
plt.bar(rtg.index, rtg.values)
plt.xlabel('Rating')
plt.ylabel('counts')
plt.show()


# In[55]:


books[['ISBN']].isnull().sum()


# In[56]:


books[['ISBN']].duplicated().any().sum()


# In[57]:


ratings[['isbn']].isnull().sum()


# In[58]:


ratings[['isbn']].duplicated().any().sum()


# In[59]:


ratings[['user_id']].isnull().sum()


# In[60]:


users[['user_id']].isnull().sum()


# In[61]:


ratings[['user_id']].duplicated().any().sum()


# In[62]:


users[['user_id']].duplicated().any().sum()


# In[63]:


books.columns


# In[64]:


books.columns = books.columns.str.strip().str.lower().str.replace('-', '_')
books.head()


# In[65]:


ratings.columns


# In[66]:


users.columns


# In[67]:


# Merge the book and ratings datasets based on the common ISBN column
book_ratings_df = pd.merge(books, ratings, on='isbn')

# Merge the resulting dataframe with the users dataset based on the common user-id column
merged_df = pd.merge(book_ratings_df, users, on='user_id')




# In[68]:


merged_df


# In[69]:


merged_df.isnull().sum()


# In[70]:


merged_df.duplicated().any().sum()


# In[71]:


merged_df[['isbn']].isnull().sum()


# In[72]:


merged_df[['user_id']].isnull().sum()


# In[73]:


merged_df[['isbn']].duplicated().any().sum()


# In[74]:


merged_df[['user_id']].duplicated().any().sum()


# In[75]:


# Assuming merged_df is your DataFrame
duplicated_rows = merged_df[merged_df.duplicated(['user_id'], keep=False)]

# Display the duplicated rows
print("Duplicated Rows:")
print(duplicated_rows)


# In[76]:


duplicated_rows = merged_df[merged_df.duplicated(['user_id', 'isbn'], keep=False)]

# Display the duplicated rows
print("Duplicated Rows based on user_id and isbn:")
print(duplicated_rows)


# In[77]:


merged_df.dtypes


# In[78]:


top_user_countries = users['country'].value_counts().head(3).index.tolist()
for country in top_user_countries:
    top_5_books = book_ratings_df[book_ratings_df['user_id'].isin(users[users['country'] == country]['user_id'])]['book_title'].value_counts().head(5)
    print(f'Top 5 books in {country}:')
    print(top_5_books)


# In[79]:


for country in top_user_countries:
    top_5_books = book_ratings_df[book_ratings_df['user_id'].isin(users[users['country'] == country]['user_id'])]['book_title'].value_counts().head(5)
    plt.bar(top_5_books.index, top_5_books.values)
    plt.xticks(rotation=90)
    plt.xlabel('Book Title')
    plt.ylabel('Count')
    plt.title(f'Top 5 Books in {country}')
    plt.show()


# In[80]:


# Get the top 3 countries
top_countries = merged_df['country'].value_counts().head(3).index.tolist()

# Loop through the top 3 countries
for country in top_countries:
    # Group the merged dataset by book title and calculate the mean rating
    avg_ratings_by_book = merged_df[merged_df['country'] == country].groupby('book_title')['book_rating'].mean()
    # Get the top 5 books by average rating
    top_5_books = avg_ratings_by_book.sort_values(ascending=False).head(5)
    # Print the results
    print(f'Top 5 books by average rating in {country}:')
    print(top_5_books)


# In[81]:


print(sorted(users.age.unique()))


# In[82]:


# Defining the age bins
age_bins = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105]

# Defining the age bin labels
age_labels = ['5-15', '15-25', '25-35', '35-45', '45-55', '55-65', '65-75', '75-85', '85-95', '95-105']

# Adding a new column
merged_df['age_group'] = pd.cut(merged_df['age'], bins=age_bins, labels=age_labels)

#  top 3 countries
top_user_countries = users['country'].value_counts().head(3).index.tolist()

# Loop through the top 3 countries
for country in top_user_countries:
    # Loop through the age groups
    for age_group in age_labels:
        # Get the top 5 books read by users in the current age group and country
        top_5_books = merged_df[(merged_df['country'] == country) & (merged_df['age_group'] == age_group)]['book_title'].value_counts().head(5)
        # Print the results
        print(f'Top 5 books in {age_group} age group from {country}:')
        print(top_5_books)


# In[83]:


for country in top_user_countries:

    for age_group in age_labels:

        top_5_books = merged_df[(merged_df['country'] == country) & (merged_df['age_group'] == age_group)]['book_title'].value_counts().head(5)

        plt.figure()
        plt.bar(top_5_books.index, top_5_books.values)
        plt.xticks(rotation=90)
        plt.xlabel('Book Title')
        plt.ylabel('Count')
        plt.title(f'Top 5 Books in {age_group} age group from {country}')
        plt.show()


# In[84]:


#Code for Analyzing Rating Variations by Age Group


# In[85]:


merged_df.columns


# In[86]:


# Calculate the average rating for each age group
average_ratings_by_age = merged_df.groupby('age_group')['book_rating'].mean().reset_index()
# Plot the average ratings for each age group
plt.figure(figsize=(10, 6))
plt.bar(average_ratings_by_age['age_group'], average_ratings_by_age['book_rating'])
plt.xlabel('Age Group')
plt.ylabel('Average Rating')
plt.title('Average Rating by Age Group')
plt.show()


# This analysis is useful for understanding the reading preferences of different age groups and can be used in recommendation systems to suggest books by specific authors to the corresponding age groups. It provides valuable insights for book publishers, marketers, and retailers to target their audience effectively and offer personalized recommendations.

# In[87]:


### POPULARITY BASED RECOMMENDATION SYSTEM
book_ratings_df = pd.merge(books, ratings, on='isbn')


# In[88]:


book_ratings_df


# In[89]:


book_ratings_df.shape


# In[90]:


num_rating_df = book_ratings_df.groupby('book_title').count()['book_rating'].reset_index()
num_rating_df.rename(columns = {'book_rating':'num_of_ratings'},inplace = True)
num_rating_df


# In[91]:


# Convert 'book_rating' to numeric, replacing non-numeric values with NaN
book_ratings_df['book_rating'] = pd.to_numeric(book_ratings_df['book_rating'], errors='coerce')

# Calculate the mean after handling non-numeric values
Avg_rating_df = book_ratings_df.groupby('book_title')['book_rating'].mean().reset_index()

# Rename the column to 'Avg_ratings'
Avg_rating_df.rename(columns={'book_rating': 'Avg_ratings'}, inplace=True)

# Display the resulting DataFrame
print(Avg_rating_df)


# In[92]:


popular_df = num_rating_df.merge(Avg_rating_df,on = 'book_title')
popular_df


# In[93]:


# we are only considering those books whose num_ratings are greater than 250

popular_df = popular_df[popular_df['num_of_ratings']>=250].sort_values('Avg_ratings',ascending = False).head(50)


# In[94]:


popular_df


# In[95]:


popular_df = popular_df.merge(books,on = 'book_title').drop_duplicates('book_title')[['book_title','book_author','year_of_publication','image_url_m','num_of_ratings','Avg_ratings']]


# In[96]:


popular_df


# In[97]:


merged_df.columns


# In[98]:


merged_df.isna().sum()


# In[99]:


columns_to_drop = ['image_url_s', 'image_url_m','location','city','state']
merged_df.drop(columns=columns_to_drop, inplace=True)


# In[100]:


merged_df.head()


# In[101]:


merged_df = merged_df.dropna(subset=['age', 'country','age_group'])
merged_df.isna().sum()


# In[102]:


merged_df.shape


# In[103]:


num_rating_df = merged_df.groupby('book_title').count()['book_rating'].reset_index()
num_rating_df.rename(columns = {'book_rating':'num_of_ratings'},inplace = True)
num_rating_df


# In[104]:


# Convert 'book_rating' to numeric, replacing non-numeric values with NaN
book_ratings_df['book_rating'] = pd.to_numeric(book_ratings_df['book_rating'], errors='coerce')

# Calculate the mean after handling non-numeric values
Avg_rating_df = book_ratings_df.groupby('book_title')['book_rating'].mean().reset_index()

# Rename the column to 'Avg_ratings'
Avg_rating_df.rename(columns={'book_rating': 'Avg_ratings'}, inplace=True)

# Display the resulting DataFrame
print(Avg_rating_df)


# In[105]:


popular_df = num_rating_df.merge(Avg_rating_df,on = 'book_title')
popular_df


# In[106]:


popular_df = popular_df.merge(merged_df,on = 'book_title').drop_duplicates('book_title')[['book_title','book_author','year_of_publication','user_id','age_group','country','image_url_l','num_of_ratings','Avg_ratings']]


# In[107]:


popular_df


# In[108]:


popular_df.columns


# In[109]:


# Step 1: Filter the dataset for a given user_id
user_id=int(input("Enter user_id: "))
user_data = popular_df[popular_df['user_id'] == user_id]
country = user_data['country'].values[0]
age_group = user_data['age_group'].values[0]


# In[110]:


top_5_recommendations = popular_df[(popular_df['country'] == country) & (popular_df['age_group'] == age_group)] \
    .sort_values(by='Avg_ratings', ascending=False) \
    .head(5)[['book_title', 'book_author']]

print(f'User {user_id} from {country} in the age group {age_group} may like the following books:')
print(top_5_recommendations)


# In[111]:


#POPULARITY BASED RS TAKING USER_ID AS INPUT
def popularity_recommendation(user_id, top_n=5):
    user_info = merged_df[merged_df['user_id'] == user_id].iloc[0]
    country = user_info['country']
    age_group = user_info['age_group']

    filtered_books = merged_df[(merged_df['country'] == country) & (merged_df['age_group'] == age_group)]
    avg_ratings = filtered_books.groupby('book_title')['book_rating'].mean().sort_values(ascending=False)

    user_rated_books = filtered_books[filtered_books['user_id'] == user_id]['book_title']
    avg_ratings = avg_ratings[~avg_ratings.index.isin(user_rated_books)]

    top_books = avg_ratings.head(top_n).index

    return country, age_group, top_books

user_id_input = int(input("Enter user ID: "))
user_country, user_age_group, top_books_recommendation = popularity_recommendation(user_id_input, top_n=5)

print(f"\nUser's Country: {user_country}")
print(f"User's Age Group: {user_age_group}")



print("\nTop 5 Popular Books Recommendation:")
for i, book_title in enumerate(top_books_recommendation, 1):
    book_info = merged_df[merged_df['book_title'] == book_title].iloc[0]
    print(f"{i}. Book Title: {book_title}")
    print(f"   Image URL: {book_info['image_url_l']}")


# In[112]:


merged_df.columns


# In[113]:


#COLLABERATIVE FILTERING RECOMMENDATION SYSTEM


# In[114]:


# Merge the book and ratings datasets based on the common ISBN column
book_ratings_df = pd.merge(books, ratings, on='isbn')


# In[115]:


# Merge the resulting dataframe with the users dataset based on the common user-id column
merged_df = pd.merge(book_ratings_df, users, on='user_id')


# filtering the data who(users) have rated more than 100 books

# In[116]:


merged_df.info()


# In[117]:


x = merged_df.groupby('user_id').count()['book_rating'] > 100
users_rated_above_100 = x[x].index


# In[118]:


filtered_ratings = merged_df[merged_df['user_id'].isin(users_rated_above_100)]


# In[119]:


y = filtered_ratings.groupby('book_title').count()['book_rating'] >= 50
famous_books = y[y].index


# In[120]:


famous_books


# In[121]:


final_ratings = filtered_ratings[filtered_ratings['book_title'].isin(famous_books)]


# In[122]:


final_ratings


# In[123]:


final_ratings.drop_duplicates()


# In[124]:


pt = final_ratings.pivot_table(index = 'user_id',columns = 'book_title',values = 'book_rating')


# the above matrix is the matrix where users are who rated more than 100 books and the books where each book has more than 50 ratings.

# In[125]:


pt.shape


# In[126]:


pt.fillna(0,inplace = True)


# In[127]:


pt


# In[128]:


from sklearn.metrics.pairwise import cosine_similarity


# In[129]:


similarity_scores = cosine_similarity(pt)


# In[130]:


print(similarity_scores)


# In[131]:


similarity_scores.shape


# Collaborative Filtering(User - Item) for Model Deployment

# In[136]:


from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
import pandas as pd

# Assuming 'merged_df' is your merged DataFrame
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(merged_df[['user_id', 'book_title', 'book_rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD()
model.fit(trainset)

def get_recommendations(user_id, n, model, all_books):
    user_unrated_books = merged_df[merged_df['user_id'] == user_id]['book_title'].unique()
    all_books = merged_df['book_title'].unique()
    to_predict = [book for book in all_books if book not in user_unrated_books]

    test_data = [(user_id, book, 0) for book in to_predict]
    predictions = model.test(test_data)

    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

    recommended_books = [item[1] for item in top_n]

    return recommended_books


# Book recommendations for User_ID
user_id_to_recommend = int(input("Enter your User_ID: "))
number_of_recommendations = int(input("Enter the number of books you want to be recommended: "))

all_books = merged_df['book_title'].unique()

recommendations = get_recommendations(user_id_to_recommend, number_of_recommendations, model, all_books)

if recommendations:
    print(f"\nTop {number_of_recommendations} book recommendations for User_ID: {user_id_to_recommend}")
    for i, book in enumerate(recommendations, 1):
        print(f"{i}. {book}")


# In[ ]:


# Create a Pickle File
import pickle
pickle_out = open("model.pkl","wb")
pickle.dump(model,pickle_out)
pickle_out.close()


# Model Evaluation
# 

# In[ ]:


from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
import pandas as pd

# Assuming 'merged_df' is your merged DataFrame
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(merged_df[['user_id', 'book_title', 'book_rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD()
model.fit(trainset)

def get_recommendations(user_id, n, model, all_books):
    test_data = [(user_id, book, 0) for book in all_books]
    predictions = model.test(test_data)

    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

    recommended_books = [item[1] for item in top_n]

    return recommended_books

# Book recommendations for User_ID
user_id_to_recommend = int(input("Enter your User_ID: "))
number_of_recommendations = int(input("Enter the number of books you want to be recommended: "))

all_books = merged_df['book_title'].unique()

recommendations = get_recommendations(user_id_to_recommend, number_of_recommendations, model, all_books)

if recommendations:
    print(f"\nTop {number_of_recommendations} book recommendations for User_ID: {user_id_to_recommend}")
    for i, book in enumerate(recommendations, 1):
        print(f"{i}. {book}")

