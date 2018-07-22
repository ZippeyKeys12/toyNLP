import re, json, train
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer, ChatterBotCorpusTrainer
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
chatbot = ChatBot(
    "Orderer",
    preprocessors=["chatterbot.preprocessors.clean_whitespace"],
    trainer="chatterbot.trainers.ChatterBotCorpusTrainer",
    logic_adapters=[
        {"import_path": "chatterbot.logic.BestMatch"},
        {
            "import_path": "chatterbot.logic.LowConfidenceAdapter",
            "threshold": 0.65,
            "default_response": "I am sorry, can you try saying that in another way?",
        },
    ],
)
if not os.path.isfile("db.sqlite3"):
    chatbot.train(
        "chatterbot.corpus.english.ai",
        "chatterbot.corpus.english.botprofile",
        "chatterbot.corpus.english.computers",
        "chatterbot.corpus.english.conversations",
        "chatterbot.corpus.english.food",
        "chatterbot.corpus.english.greetings",
    )
    chatbot.set_trainer(ListTrainer)
    Triggers = ["I would like", "I want", "Can I have", "Can I get", "Get me", "I need"]
    ordering = ["What's the special", ";Show:Special;"]
    with open("../Data/foods.json", "r") as Data:
        Foods = json.load(Data)
    for Category in Foods:
        for Food in Foods[Category]:
            Food = Food["name"]
            if Food == "burger" or Food == "hotdog":
                AFood = "a " + Food
            else:
                AFood = Food
            for Trigger in Triggers:
                ordering.extend(
                    [Trigger + " " + Food, "Ordered " + AFood + ";Order:" + Food + ";"]
                )
    chatbot.train(ordering)
Parse = re.compile(";(\\w+):(\\w+);")


def parse_text(cmd):
    confidence, response=train.main(cmd)
    if confidence>90:
        responder="Ordered "
        command=["Order"]
        for i in response:
            command.extend(i)
            responder+=i[0]
        return responder, command
    cmd = ps.stem(cmd)
    response = str(chatbot.get_response(cmd))
    Command = Parse.search(response)
    if Command:
        response = response.replace(Command.group(0), "")
        Command = [Command.group(1), Command.group(2), 1]
    return response, Command


if __name__ == "__main__":
    while True:
        print(parse_text(input("What? ")))
