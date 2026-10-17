import re
import string
import emoji
import contractions


def remove_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)

  
def remove_mentions(text):
    return re.sub(r"@\w+", "", text)


def handle_hashtags(text, option="remove"):
    if option == "remove":
        return re.sub(r"#(\w+)", r"\1", text)
    if option == "empty":
        return re.sub(r"#\w+", "", text)
    return text


def clean_text(text, hashtag_option="remove", emoji_option="remove"):
    text = str(text).lower()
    text = remove_urls(text)
    text = remove_mentions(text)
    text = handle_hashtags(text, hashtag_option)

    if emoji_option == "remove":
        text = emoji.replace_emoji(text, replace="")
    elif emoji_option == "text":
        text = emoji.demojize(text)

    text = contractions.fix(text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()
