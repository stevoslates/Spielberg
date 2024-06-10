# Speilberg #
## Speilberg is a Chatbot built using Retreival Augmented Generation (RAG). It uses my data from LetterBoxd (Movie Review Website and Social Media App) to essentially create my movie brain in a chatbot. ##

During creation, I wanted to experiement with different RAG pipelines, as this is the first time I had used strucutred data for RAG. Speilberg allows you to choose between a normal pipeline and a self query pipeline.

#### Normal Pipeline ####
Here I merged the structured data into one column, transformed it into JSON and used the standard retrevier with the vectorDB of the data. 

#### Self Query Pipeline ####
Uses the Self Query retreival method, where the some attributes from the data are transformed into the metadata and metadata functions provide insights into them to aid retrieval and responses. Also stores in vectorDB.
This seems to create better responses than normal pipeline

#### Next Steps ####
* More Data (Genres, Directors, etc) for better conversation.
* Chat Memory
