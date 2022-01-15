import pickle
from preprocess import ek_extra_preprocess
from tqdm import tqdm

all_posts = set()
user_id_2_post_list = pickle.load(open("../../data/non-graph/user_id_posts_idx_list.pkl", "rb"))

for user_id in tqdm(user_id_2_post_list):
    for post_id in user_id_2_post_list[user_id]:
        all_posts.add(post_id)
print(len(all_posts))
all_post_ids = sorted(list(all_posts))

gab_data = pickle.load(open("../../../sumegh/GabData/Final_Posts.pkl", "rb"))

post_id_2_text = dict()
for post_id in tqdm(all_post_ids):
    text = gab_data[post_id]['post_body']
    tokens = ek_extra_preprocess(text, {'include_special': False, 'bert_tokens': False}, None)
    post_id_2_text[post_id] = ' '.join(tokens)

pickle.dump(post_id_2_text, open("../../data/non-graph/posts_id_2_text.pkl", "wb"))
