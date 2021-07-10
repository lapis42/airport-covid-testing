"""
Script for COVID test reservation
"""

import time, sched
import keyring

from selenium.webdriver import Firefox
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.common.exceptions import TimeoutException

from utils import tgprintf

TARGET_TIME = time.mktime(time.strptime('Sat Jul 10 04:59:00 2021'))


def main():
    print('{}: running...'.format(time.ctime(time.time())))

    browser = Firefox()
    browser.get('https://www.airport.kr/ap/ko/svc/covidReservation.do')
    wait = WebDriverWait(browser, 20, poll_frequency=1)

    try:

        # start login
        wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "btn-type-normal")))
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

        # date
        while True:

            wait.until(EC.presence_of_element_located((By.ID, "DAY")))
            date = browser.find_element_by_id('DAY')
            date_btn = browser.find_element_by_id('getTimeListBtn')
            date.clear()

            my_date = keyring.get_password('covid', 'date')
            date.send_keys([my_date, Keys.TAB])

            if date.get_attribute('value') == my_date:

                date_btn.click()

            try:

                WebDriverWait(browser, 3).until(EC.alert_is_present())
                alert = browser.switch_to.alert
                time.sleep(3)
                alert.accept()
                time.sleep(1)
                browser.refresh()

            except TimeoutException:
                break

        # select time
        t = Select(browser.find_element_by_id('TIME'))
        n_options = lambda x: len(t.options) > 10
        wait.until(n_options)
        found = False
        for x in t.options:
            if '가능' in x.text and int(x.text[:2]) >= 13:
                found = True
                t.select_by_visible_text(x.text)
                break

        if not found:
            tgprintf('Available time not found...')
            return

        # infos
        flt = browser.find_element_by_id('FLT_DATE')
        flt.clear()
        flt.send_keys([MY_DATE, Keys.TAB])
        assert flt.get_attribute('value') == MY_DATE
        flt_hour = Select(browser.find_element_by_id('FLT_HOUR'))
        flt_min = Select(browser.find_element_by_id('FLT_MIN'))
        flt_hour.select_by_visible_text('17')
        flt_min.select_by_visible_text('15')
        arr = browser.find_element_by_id('ARR_CODE')
        arr.clear()
        arr.send_keys(['미국', Keys.TAB])
        terminal = Select(browser.find_element_by_id('TERMINAL_ID'))
        terminal.select_by_value('T1')

        # test type
        inspection = Select(browser.find_element_by_id('INSPECTION_TYPE'))
        inspection_list = [x.text for x in inspection.options]
        if '항원검사' in inspection_list:
            inspection.select_by_visible_text('항원검사')
        elif 'PCR 검사' in inspection_list:
            inspection.select_by_visible_text('PCR 검사')

        resv_count = Select(browser.find_element_by_id('RESV_COUNT'))
        resv_count.select_by_visible_text('1')

        # captcha

        # agreement
        agree = browser.find_element_by_id('AGREE04')
        agree_btn = browser.find_element_by_class_name('checkbox-type2-label')
        if not agree.is_selected():
            agree_btn.click()

        tgprintf('Successful!')

    except Exception as e:
        tgprintf('Error!')
        print(e)


if __name__ == "__main__":
    print('{}: standby...'.format(time.ctime(time.time())))
    s = sched.scheduler(time.time, time.sleep)
    s.enterabs(TARGET_TIME, 1, main)
    s.run()
