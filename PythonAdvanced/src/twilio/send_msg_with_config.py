import configobj
from twilio.rest import TwilioRestClient

'''
Ref:
http://www.blog.pythonlibrary.org/2014/09/23/python-101-how-to-send-smsmms-with-twilio/

=> Good info on reading a config file usning configobj package
'''

def send_mms(msg, to, img):
    """"""
    cfg = configobj.ConfigObj("/Users/dbiswas/Documents/SourceCode/python/python-projects/PythonAdvanced/src/resources/conf/twilio.cfg")
    account_sid = cfg["twilio"]["account_sid"]
    auth_token = cfg["twilio"]["auth_token"]
    twilio_number = cfg["twilio"]["twilio_number"]

    client = TwilioRestClient(account_sid, auth_token)

    message = client.messages.create(body=msg,
                                     from_=twilio_number,
                                     to=to,
                                     MediaUrl=img
                                     )

    print message.sid

def send_sms(msg, to):

    cfg = configobj.ConfigObj("/Users/dbiswas/Documents/SourceCode/python/python-projects/PythonAdvanced/src/resources/conf/twilio.cfg")
    account_sid = cfg["twilio"]["account_sid"]
    auth_token = cfg["twilio"]["auth_token"]
    twilio_number = cfg["twilio"]["twilio_number"]

    client = TwilioRestClient(account_sid, auth_token)

    message = message = client.messages.create(body=msg, to=to, from_=twilio_number)

    print(message.sid)


if __name__ == "__main__":
    msg = "Hello from Python!"
    to = "+16027486894"
    img = "http://www.website.com/example.jpg"
    #Send MMS
    #send_mms(msg, to=to, img=img)

    #Send SMS
    msg = "Hey...experiment working..."
    send_sms(msg, to)
