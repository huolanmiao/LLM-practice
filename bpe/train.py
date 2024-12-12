from bpe import Tokenizer
tokenizer = Tokenizer()
with open('./manual.txt', 'r', encoding='utf-8') as file:
    text = file.read()
# tokenizer.train(text, 1024)

tokenizer.load("./manual.model")
with open('./train.txt', 'w', encoding='utf-8') as file:
    file.write(f"encoded training text : {tokenizer.encode(text)}")
    file.write(f"encode and decode match: {text == tokenizer.decode(tokenizer.encode(text))}")
    
# tokenizer.save("manual")