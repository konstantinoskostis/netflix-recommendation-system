import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')


class Preprocessor:
    """Preprocess a pandas data frame

    This class will read a pandas data frame and will process the description
    field of the given data frame.

    Attributes
        df: A pandas data frame
    """

    def __init__(self, df):
        self.updated_df = df.copy()
        self.stopwords = stopwords.words('english')
        self.stemmer = PorterStemmer()

    def preprocess(self, column='Description'):
        """Preprocess the data frame.

        Args:
            column: The name of the text column to pre-process.
        Returns:
            None
        """
        self.updated_df['processed_description'] = self.updated_df[column].\
            apply(self.clean)

    def clean(self, text):
        """Clean text via NLTK

        This method cleans the given text by performing the following steps:
        - lower case
        - tokenization (using word_tokenize from NLTK)
        - filter punctuation tokens
        - stem

        Args:
            text: The text to clean

        """
        tokens = nltk.word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in self.stopwords]
        tokens = list(filter(lambda token: token not in string.punctuation,
                             tokens))
        tokens = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(tokens)
