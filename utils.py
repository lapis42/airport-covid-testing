import telegram
import keyring

def tgprintf(msg):
    bot = telegram.Bot(token=keyring.get_password('telegram', 'token'))
    chat_id = keyring.get_password('telegram', 'id')
    bot.send_message(chat_id=chat_id, text=msg)
