# possible methoeds:
- zero-shot classification using BERTopic
- - https://maartengr.github.io/BERTopic/getting_started/zeroshot/zeroshot.html#example
- - set of predefined topics that the model first checks
- - if none of the rpedefined topics fit, the model finds a new topic of the text
- zero-shot through chatGPT
- 

# possible models to use
- SEZNAM models: https://github.com/seznam/czech-semantic-embedding-models?tab=readme-ov-file
- chatGPT (requires API key)
- SBert models: https://www.sbert.net/docs/pretrained_models.html
- - multi-lingual: paraphrase-multilingual-mpnet-base-v2, paraphrase-multilingual-MiniLM-L12-v2, distiluse-base-multilingual-cased-v2
- BERTopic ZeroShotClassification: https://maartengr.github.io/BERTopic/api/representation/zeroshot.html#bertopic.representation._zeroshot.ZeroShotClassification
- facebook/bart-large-mnli: https://huggingface.co/facebook/bart-large-mnli
- - works with Czech too
- - 0.509 binary cross entropy loss
- zero-shot topic modeling: https://maartengr.github.io/BERTopic/getting_started/zeroshot/zeroshot.html
- Czert-B-base-cased-long-zero-shot: https://huggingface.co/UWB-AIR/Czert-B-base-cased-long-zero-shot
- SlavicBert: https://huggingface.co/DeepPavlov/bert-base-bg-cs-pl-ru-cased