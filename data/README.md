# Datasets

In this project, we collect 1 top news dataset from googole news and leverage 3 news benchmark datasets available online. Here is the discription of the dataset.

## GoogleNews Top story.
The GoogleNews dataset is collected between November 25, 2024 and December 8, 2024, from Google News' top stories in the US. It includes all articles on the full coverage page of the top 10 stories each day, with each story containing 
approximately 50 articles. We collect the [Google Top news page](https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFZxYUdjU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US%3Aen) profileby [SerpAPI](https://serpapi.com/google-news-api), which contians ranked stories( a set of theme-alike articles) every day. We provide the following data: 
* rankedaw_data/Googlenews/ 

    * Topstories: this folder contains top-10 stories (p1-p10) every day during the data collection period. 

    * Topstory_profile: this folder contains the profiles of Google new Top story page, which contains approximately 50 stories each day.


## Covidnews  
The Covid news dataset, provided by [AYLIEN](https://aylien.com/), consists of English news articles from 440 global sources published between November 2019 and July 2020. All articles are related to COVID-19. Application of the access if provide by [link](https://aylien.com/resources/datasets/coronavirus-dataset).

## NELA-2022
The [NELA](https://github.com/MELALab/nela-gt) dataset contains 1\,778\,361 articles from 361 global sources, spanning January 2022 to December 2022.  

## WCEP
Last, the WCEP dataset is a benchmark news dataset collected from the Wikipedia Current Event Portal and the 
Common Crawl Archive; it includes ground-truth story labels for each article. The access is provide by [link](https://github.com/complementizer/wcep-mds-dataset).


## MBFC data

Media Bias Fact Checking data is available from the website [MBFC](https://mediabiasfactcheck.com/)