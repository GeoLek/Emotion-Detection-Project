# Import necessary packages

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize  # Tokenization (split texts into individual word or tokens)
from nltk.corpus import stopwords  # Remove stopwords (e.g., 'the','is')
import string  # Handle strings
import re  # Regular expressions - string manipulation


# Ensure NLTK components are downloaded

nltk.download('punkt')
nltk.download('stopwords')

#testing