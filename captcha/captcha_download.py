import os, time
import keyring

from selenium.webdriver import Firefox
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


browser = Firefox()
browser.get('https://www.airport.kr/ap/ko/svc/covidReservation.do')
wait = WebDriverWait(browser, 20, poll_frequency=1)

# start login
wait.until(EC.presence_of_element_located((By.CLASS_NAME, "btn-type-normal")))
login = browser.find_element_by_class_name('btn-type-normal')
login.click()

# enter login
wait.until(EC.visibility_of_element_located((By.ID, "LOGIN_USER_ID")))
input_id = browser.find_element_by_id('LOGIN_USER_ID')
input_pass = browser.find_element_by_id('LOGIN_USER_PWD')
login_press = browser.find_element_by_class_name('layer-btn-login')
input_id.send_keys([keyring.get_password('covid', 'id'), Keys.TAB])
input_pass.send_keys([keyring.get_password('covid', 'pass'), Keys.TAB])
login_press.click()

# captcha
wait.until(EC.visibility_of_element_located((By.ID, 'captchaImg')))
captcha = browser.find_element_by_id('captchaImg')

data_path = './data'
if not os.path.exists(data_path):
    os.mkdir(data_path)


prev_img = captcha.screenshot_as_png
for i in range(10000):
    browser.find_element_by_class_name('capcha.reflash').click()
    while True:
        time.sleep(1)
        img = captcha.screenshot_as_png
        if img != prev_img:
            prev_img = img
            break
    with open(os.path.join(data_path, '{}.png'.format(captcha.get_attribute('src').split('?')[1])), 'wb') as f:
        f.write(img)

browser.close()
