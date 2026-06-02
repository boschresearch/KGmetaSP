import pandas as pd
import numpy as np
import torch
from nltk.corpus import stopwords
import gensim.corpora as corpora
import gensim

import re
import base64, io
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

from utils import (
    get_relevant_relations,
    POTENTIAL_TEXT_TYPES,
    Data,
    URI_PREFIX,
    IMAGE_TYPES,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# currently not applicable since all bins are part of triples where adjacent bins are described
def delete_empty_bin_types(data: Data, num_bins: int) -> Data:
    """
    Delete empty bin types from the dataset.

    This method takes a Data object representing a KG and a number of relevant bins. 
    It identifies and deletes any empty bin types from the dataset. 
    The updated Data object with the empty bin types removed is returned.

    Args:
        data (Data): The Data object representing the dataset.
        num_bins (int): The number of bins to check.

    Returns:
        Data: The updated Data object with empty bin types deleted.

    """

    to_delete = []
    for d in data.i2e[-num_bins:]:
        filtered = data.triples[data.triples[:, 2] == data.e2i[d]]
        if len(filtered) == 0:
            to_delete.append(data.e2i[d])

    if len(to_delete) != 0:
        print(f"deleting relations {to_delete}, since no occurences are given")
        new_e2i = {}
        new_i2e = []

        for i in range(len(data.i2e)):
            if i not in to_delete:
                nt = torch.tensor(len(new_i2e), dtype=torch.int32)
                it = torch.tensor(i, dtype=torch.int32)
                new_e2i[data.i2e[it]] = nt
                new_i2e.append(data.i2e[it])
                
        # apply new mapping for triples
        for t in data.triples:
            t[0] = new_e2i[data.i2e[t[0]]]
            t[2] = new_e2i[data.i2e[t[2]]]

        # update metedata
        data.num_entities = len(new_i2e)

        data.i2e = new_i2e
        data.e2i = new_e2i
        print("done deleteing")
        print(data.name)
    return data


def _get_stopword_list(
    languages=["dutch", "spanish", "french", "portuguese", "english"]
):
    """
    Get a list of stopwords for the specified languages.

    This method returns a list of stopwords for the specified languages. By default, stopwords for Dutch, Spanish, French,
    Portuguese, and English are included in the list. Additional languages can be specified by providing a list of language
    names as the 'languages' parameter.

    Args:
        languages (List[str], optional): The list of languages for which to retrieve stopwords. Defaults to ["dutch",
            "spanish", "french", "portuguese", "english"].

    Returns:
        List[str]: The list of stopwords for the specified languages.

    """
    stopword_list = []
    for language in languages:
        stopword_list.extend(stopwords.words(language))
    return stopword_list


def _remove_stopwords(word_array, stopword_list):
    """
    Remove stopwords from the given word array.

    This method removes stopwords from the given word array by filtering out words that are present in the stopword list.

    Args:
        word_array (List[str]): The array of words to remove stopwords from.
        stopword_list (List[str]): The list of stopwords to be removed.

    Returns:
        List[str]: The word array with stopwords removed.

    """
    return [word for word in word_array if word not in stopword_list]


def LDA_topic_assignment(
    data:Data,
    num_topics:int=10,
    min_mean_word_count:int=3,
)-> Data:
    """
    GDA approach, performing topic assignment using Latent Dirichlet Allocation (LDA) on the given KG.

    This method takes the KG and performs topic assignment of text literals using Latent Dirichlet Allocation (LDA).
    It assigns topics to the data based on the specified number of topics (num_topics) and minimum mean word count
    (min_mean_word_count) threshold for a predicate bound set of documents to be processed.

    Args:
        data (Data): The input KG for topic assignment.
        num_topics (int): The number of topics to be assigned. Defaults to 10.
        min_mean_word_count (int): The minimum mean word count threshold. Defaults to 3.

    Returns:
        Data: The updated data with assigned topics.

    """
    relevent_relations = get_relevant_relations(
        data, relevant_types=POTENTIAL_TEXT_TYPES
    )

    for b in range(num_topics):
        o = (f"{URI_PREFIX}entity#topic{b}", f"{URI_PREFIX}datatype#topics")
        new_id = len(data.i2e)
        data.e2i[o] = new_id
        data.i2e.append(o)
        data.num_entities += 1

    stopword_list = _get_stopword_list()
    for r in relevent_relations:
        df = pd.DataFrame(
            data.triples[data.triples[:, 1] == r], columns=["s", "p", "o"]  # type: ignore
        )
        df["text"] = df["o"].apply(lambda t: data.i2e[t][0])
        df["type"] = df["o"].apply(lambda t: data.i2e[t][1])

        # delete "none type" polygons
        df["text"] = df["text"].apply(
            lambda t: "" if re.match("(MULTIPOLYGON|POLYGON)", t) else t
        )
        mean_num_words = df["text"].str.count(r"([\w\:\.\/]{3,})").mean()
        # only process if mean word count is higher than trashold
        if mean_num_words > min_mean_word_count:
            p = f"{URI_PREFIX}predicat#topics{r}"
            new_id = len(data.i2r)
            data.r2i[p] = new_id
            data.i2r.append(p)
            data.num_relations += 1

            df["text_preprocessed"] = df["text"].apply(
                lambda t: gensim.utils.simple_preprocess(t, deacc=True)
            )
            df["text_preprocessed"] = df["text_preprocessed"].apply(
                lambda t: _remove_stopwords(t, stopword_list=stopword_list)
            )

            data_words = df.text_preprocessed.values.tolist()
            # data_words = list(sent_to_words(text_data))
            print(data_words[:1][0][:30])
            # Create Dictionary
            id2word = corpora.Dictionary(data_words)
            # Term Document Frequency
            corpus = [id2word.doc2bow(text) for text in data_words]
            print(corpus[:1][0][:30])
            num_topics = 10
            lda_model = gensim.models.LdaMulticore(
                corpus=corpus,
                iterations=100,
                id2word=id2word,
                num_topics=num_topics,
                random_state=42,
            )
            df["vector"] = df["text_preprocessed"].apply(
                lambda t: id2word.doc2bow(t)
            )
            df["topics"] = df["vector"].apply(
                lambda t: [
                    x[0]
                    for x in sorted(
                        lda_model.get_document_topics(t),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:3]
                    if x[1] > 0.1 #type: ignore
                ]
            )
            for i in range(num_topics):
                df[f"topic_{i}"] = df["topics"].apply(
                    lambda t: True if i in t else False
                )

            for i in range(num_topics):
                sub = df[df[f"topic_{i}"] == True]
                if len(sub) > 0:
                    sub_df = torch.zeros(len(sub), 3, dtype=torch.int32)
                    sub_df[:, 0] = torch.tensor(
                        sub.s.tolist(), dtype=torch.int32
                    )
                    # torch.full((len(sub),1),data.r2i[f'{URI_PREFIX}predicat#topics{9}'], dtype=torch.int32)
                    sub_df[:, 1] = data.r2i[f"{URI_PREFIX}predicat#topics{r}"]
                    sub_df[:, 2] = data.e2i[
                        f"{URI_PREFIX}entity#topic{i}",
                        f"{URI_PREFIX}datatype#topics",
                    ]
                    data.triples = torch.cat((data.triples, sub_df), 0)
    return data


def _is_image(b64)-> bool:
    """
    Check if the given base64 string represents an image.

    This method checks if the given base64 string represents an image by performing a validation on the string format
    or by decoding the base64 string and checking if it can be loaded as an image.

    Args:
        b64 (str): The base64 string to check.

    Returns:
        bool: True if the given string represents an image, False otherwise.

    """
    try:
        base64.urlsafe_b64decode(b64)
    except:
        return False
    return True


def _get_image(b64):
    """
    Decode the given base64 string and return the corresponding image.

    This method decodes the given base64 string using the `urlsafe_b64decode` function from the `base64` module.
    It then tries to open the decoded data as an image using the `Image.open` function from the `PIL` (Pillow) library.
    If the decoding or opening process fails, it returns a default blank image.

    Args:
        b64 (str): The base64 string representing the image.

    Returns:
        Image: The decoded image if successful, or a default blank image otherwise.

    """
    try:
        base64.urlsafe_b64decode(b64)
    except:
        print(f"Could not decode b64 string {b64}")
        return Image.new("RGB", (1, 1))
    try:
        return Image.open(io.BytesIO(base64.urlsafe_b64decode(b64)))
    except:
        return Image.new("RGB", (1, 1))



def VGG_image_classification(data:Data, **kwargs)->Data:
    """
    GDA approach, performing image classification using the VGG model on the given data.

    This method performs image classification using a pre-trained VGG (Visual Geometry Group) model on the given data.


    Args:
        data (Data): The input data for image classification.
        **kwargs: Additional keyword arguments for customization.

    Returns:
        Data: The result of the image classification.

    """
    relevant_relations = get_relevant_relations(data, IMAGE_TYPES)
    df = pd.DataFrame(
        data.triples[
            torch.isin(data.triples[:, 1], torch.tensor(relevant_relations))  #type: ignore
        ],
        columns=["s", "p", "o"],
    )
    vgg = models.vgg16(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = vgg.to(device)
    vgg.eval()

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Resize(256),  # Resize the image to 256x256 pixels
        ]
    )

    df["b64"] = df["o"].apply(lambda x: data.i2e[x][0])
    df["is_image"] = df["b64"].apply(lambda x: _is_image(x))
    df = df[df["is_image"] == True]
    df["image"] = df["b64"].apply(lambda x: _get_image(x)) #type: ignore
    df["class"] = df["image"].apply(
        lambda x: vgg(preprocess(x).unsqueeze(0).to(device))
        .squeeze(0)
        .softmax(0)
        .argmax()
        .item()
    )
    print(df["class"])

    for pred in df[
        "p"
    ].unique():  # use this over relation since loading image can subset df
        sub_df = df[df["p"] == pred]
        p = f"{URI_PREFIX}predicat#img-class-{pred}"
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1
        for c in sub_df["class"].unique():
            o = (
                f"{URI_PREFIX}entity#img-class-{c}-{pred}",
                f"{URI_PREFIX}datatype#img-class",
            )
            new_id = len(data.i2e)
            data.e2i[o] = new_id
            data.i2e.append(o)
            data.num_entities += 1

        sub_df["new_o"] = sub_df["class"].apply(
            lambda c: data.e2i[
                (
                    f"{URI_PREFIX}entity#img-class-{c}-{pred}",
                    f"{URI_PREFIX}datatype#img-class",
                )
            ]
        )
        sub_df["new_p"] = sub_df["class"].apply(
            lambda f: data.r2i[f"{URI_PREFIX}predicat#img-class-{pred}"]
        )
        ten = torch.tensor(
            sub_df[["s", "new_p", "new_o"]].values.astype(np.int32),
            dtype=torch.int32,
        )
        data.triples = torch.cat((data.triples, ten), 0)
    return data
