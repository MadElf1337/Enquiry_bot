import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random

intents = json.loads(open('intents.json', encoding="utf8").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')

sentence = " "  #using an arbitrary variable as an example and for further usage in functions


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


bot_name = ""           #add any random bot name here
import tkinter as tk


class Chat:
    def __init__(self):
        self.window = tk.Tk()
        self._setup_window()

    def _setup_window(self):
        self.window.title("enquiry_bot")
        self.window.resizable(width=False, height=False)
        _width, _height = 400, 550
        self.window.configure(width=_width, height=_height, bg='grey')

        # Center the window
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        # For left-align
        left = (screen_width / 2) - (_width / 2)
        # For right-align
        top = (screen_height / 2) - (_height / 2)
        # For top and bottom
        self.window.geometry('%dx%d+%d+%d' % (_width, _height,
                                              left, top))

        top_label = tk.Label(self.window, bg='grey', fg='black',
                             text='MMCOE Chat Bot', pady=6, font='12', )
        top_label.place(relwidth=1)

        divide_line = tk.Label(self.window, width=400, bg='green')
        divide_line.place(relwidth=1, rely=0.06, relheight=0.012)

        # text instance variable
        self.text_widget = tk.Text(self.window, width=20, height=2,
                                   bg='grey13', padx=5, pady=5, fg='white', wrap='word',
                                   font='Courier 12')
        self.text_widget.place(relheight=0.745, relwidth=0.97, rely=0.07)
        self.text_widget.configure(state=tk.DISABLED, cursor='arrow')

        scrollbar = tk.Scrollbar(self.window)
        scrollbar.place(relheight=0.744, relx=0.97, relwidth=0.03, rely=0.07)
        scrollbar.configure(command=self.text_widget.yview)
        self.text_widget.config(yscrollcommand=scrollbar.set)

        bottom_label = tk.Label(self.window, bg='grey', height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # message instance variable
        self.message_entry = tk.Entry(bottom_label, bg='white', font='12')
        self.message_entry.place(relwidth=0.75, relheight=0.05, rely=0.012,
                                 relx=0.011)
        self.message_entry.focus()  #focus app window post launch
        self.message_entry.bind("<Return>", self._on_return_pressed)

        send_button = tk.Button(bottom_label, text='Send', width=20,
                                bg='green', command=lambda: self._on_return_pressed(None),
                                font='12', )
        send_button.place(relx=0.78, rely=0.012, relheight=0.05, relwidth=0.20)

    def _on_return_pressed(self, event=None):
        global _message
        _message = self.message_entry.get()
        self._insert_message(_message, 'You')
        # self.window.after(1500, self._insert_answer)  # delay the answer

    def _insert_message(self, _message, sender):
        if not _message:
            return

        self.message_entry.delete(0, tk.END)
        msg1 = f"{sender}: {_message}\n\n"
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.insert(tk.END, msg1)
        self.text_widget.configure(state=tk.DISABLED)

        msg2 = f"{bot_name}: {chatbot_response(_message)}\n\n"
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.insert(tk.END, msg2, 'bot')
        self.text_widget.configure(state=tk.DISABLED)

        self.text_widget.see(tk.END)

    def run(self):
        self.window.mainloop()


if __name__ == '__main__':
    chat = Chat()
    chat.run()


