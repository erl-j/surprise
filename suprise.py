#%%
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib
from IPython.display import HTML, display
import torch
import seaborn as sns
import matplotlib.pyplot as plt

#%%
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", return_dict=True)

# %%
def colorize(words, values):
    # Ensuring values are within the range [0, 1]
    values = (values - values.min()) / (values.max() - values.min())

    # Generating HTML parts for each word
    html_string = ""
    for word, value in zip(words, values):
        color = f"rgb({255 * (1 - value)}, {255 * (1 - value)}, {255})"
        html_string += f"<span style='color: {color}'>{word}</span> "

    # Displaying the colored words
    display(HTML(html_string))

#%%
a = "The alphabet is a, b, c, d, d, d, e."


def analyze_text(a):
    input_ids = tokenizer.encode(a, return_tensors="pt")
    out = model(input_ids, return_dict=True)
    logits = out.logits

    logits = logits[0, :-1, :]
    input_ids = input_ids[0, 1:]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    probs = torch.softmax(logits, dim=-1)
    log_probs = -torch.log_softmax(logits, dim=-1)
    # get log_probs of input_ids
    ic = log_probs[range(len(tokens)), input_ids]
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)

    aic = ic / entropy

    aic = aic.detach().numpy()
    ic = ic.detach().numpy()
    return tokens, ic, aic

tokens, ic, aic = analyze_text(a)


def plot_token_values(tokens, values):
    sns.set_theme(style="whitegrid")
    # make plot bigger
    plt.rcParams["figure.figsize"] = (10, 10)
    # tight layout
    matplotlib.rcParams.update({'figure.autolayout': True})
    ax = sns.barplot(x=range(len(tokens)), y=values)
    ax.set_xticklabels(tokens, rotation=45)
    plt.show()

plot_token_values(tokens, ic)
plot_token_values(tokens, aic)
#%%

text = '''
The rest of the team went down to the thai restaurant to get lunch.
I stayed behind to finish my work. I was almost done when I heard a loud noise.
I knew immedately it was a bomb. I ran out of the building as fast as I could.
Once I reached the end of the block, I tuned around. The building was gone.
In it's place was a house. I looked at the house number. It was 1230000000.
'''

tokens, ic, aic = analyze_text(text)

colorize(tokens, ic)
colorize(tokens, aic)

# %%

text = '''heads, tails, heads, tails, heads, heads'''.strip()

tokens, ic, aic = analyze_text(text)

plot_token_values(tokens, ic)
plot_token_values(tokens, aic)

colorize(tokens, ic)
colorize(tokens, aic)

# %%

