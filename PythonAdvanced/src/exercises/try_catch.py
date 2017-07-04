d = dict(weather = "clima", earth = "terra", rain = "chuva")

def translate(word):
    try:
        return d[word]
    except KeyError:
        return "Word is not found"

word = raw_input("Please enter an english word : ")
print "word         : ", word
print "word.strip() : ", word.strip()
print translate(word.strip())
