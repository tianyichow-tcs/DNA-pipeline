# DNA-pipeline
The diverse news annotation (DNA) pipeline enhance news datasets by providing news stories, topic modeling, entity extraction, stance detection.


### News story discovery
News story discovery is implemented with [UStory](https://github.com/cliveyn/USTORY).  The method considers the set of articles $A$ as a time-based stream and it clusters the articles based on their similarity, as computed in a sliding time window of a given size. 

### Stance detection
Stance detection consider the clasification that takes article $a$ and a target entity $e$ as input, and predict the stance in a pre-defined category $\{\text{in-favor,\text{neutral},\text{against}}\}$. We prompt the LLM to run the stance detection task. In particular, we consider only the political entity as target, and extract the entity with [NER](https://huggingface.co/dslim/bert-base-NER), and then match with Wikidata. 

### Google News
We collected 2-weeks full coverage data from [Google TopNews](https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFZxYUdjU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US%3Aen). Each story consist a set of news articles from different news outlets. The data is available at **/data/rawdata/Googlenews**