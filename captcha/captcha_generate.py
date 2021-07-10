from PIL import Image, ImageDraw, ImageFont, ImageFilter
import csv
import numpy as np
import os

FONT_PATH = '/usr/share/fonts/opentype'
FONT_NAME = ['noto']
font_list = []
for i in FONT_NAME:
    temp = os.listdir(os.path.join(FONT_PATH, i))
    for j in temp:
        if j.endswith('.ttc') and 'Thin' not in j and 'Light' not in j:
            font_list.append(os.path.join(FONT_PATH, i, j))
n_font = len(font_list)


def save_bg():
    img = Image.open('./data/1625785903991.png')
    bg = img.crop((0, 1, 250, 2)).resize((250, 50))
    bg.save('./bg.png')


def make_bezier(xys):
    # xys should be a sequence of 2-tuples (Bezier control points)
    n = len(xys)
    combinations = pascal_row(n - 1)

    def bezier(ts):
        # This uses the generalized formula for bezier curves
        # http://en.wikipedia.org/wiki/B%C3%A9zier_curve#Generalization
        result = []
        for t in ts:
            tpowers = (t**i for i in range(n))
            upowers = reversed([(1 - t)**i for i in range(n)])
            coefs = [
                c * a * b for c, a, b in zip(combinations, tpowers, upowers)
            ]
            result.append(
                tuple(
                    sum([coef * p for coef, p in zip(coefs, ps)])
                    for ps in zip(*xys)))
        return result

    return bezier


def pascal_row(n, memo={}):
    # This returns the nth row of Pascal's Triangle
    if n in memo:
        return memo[n]
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n // 2 + 1):
        # print(numerator,denominator,x)
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n & 1 == 0:
        # n is even
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result))
    memo[n] = result
    return result


def generate(n_captcha, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    numbers = np.random.randint(0, 10, (n_captcha, 6))
    location_x = np.random.randint(
        0, 5, (n_captcha, 6)) + np.arange(6).reshape(-1, 6) * 17 + 12
    location_y = np.random.randint(0, 5, (n_captcha, 6)) - 10
    angle = np.random.randint(-5, 5, (n_captcha, 6))
    font = np.random.randint(0, n_font, (n_captcha, 6))
    size = np.random.randint(35, 40, (n_captcha, 6))
    line_start = np.random.randint(0, 50, (n_captcha, 2))
    line_mid_x = np.random.randint(25, 225, (n_captcha, 2))
    line_mid_y = np.random.randint(10, 40, (n_captcha, 2))
    line_end = np.random.randint(0, 50, (n_captcha, 2))

    for i in range(n_captcha):
        print(i)
        fn = os.path.join(save_path, '{:05}.png'.format(i))

        img = Image.open('./bg.png')

        for j in range(6):
            text = Image.new('RGBA', (40, 45))
            textdraw = ImageDraw.Draw(text)
            textdraw.text((0, 0),
                          str(numbers[i, j]),
                          font=ImageFont.truetype(font_list[font[i, j]],
                                                  size=size[i, j]),
                          fill=(0, 0, 0, 255))
            text = text.rotate(angle[i, j], expand=True)
            img.paste(text, (location_x[i, j] + 4, location_y[i, j] + 4), text)
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

        for j in range(6):
            text = Image.new('RGBA', (40, 45))
            textdraw = ImageDraw.Draw(text)
            textdraw.text((0, 0),
                          str(numbers[i, j]),
                          font=ImageFont.truetype(font_list[font[i, j]],
                                                  size=size[i, j]),
                          fill=(0, 0, 0, 255))
            text = text.rotate(angle[i, j], expand=True)
            img.paste(text, (location_x[i, j], location_y[i, j]), text)

        for j in range(2):
            xys = [(25, line_start[i, j]), (line_mid_x[i, j], line_mid_y[i,
                                                                         j]),
                   (225, line_end[i, j])]
            bezier = make_bezier(xys)
            points = bezier(np.linspace(0, 1, 10))

            linedraw = ImageDraw.Draw(img)
            for k in range(len(points) - 1):
                linedraw.line([points[k], points[k + 1]],
                              fill='black',
                              width=4)

        img = img.convert('RGB')
        img.save(fn)

    np.savetxt(os.path.join(save_path, '{}.csv'.format(save_path)),
               numbers,
               delimiter=',',
               fmt='%d')


if __name__ == "__main__":
    if not os.path.exists('bg.png'):
        save_bg()
    generate(50000, 'train')
    generate(10000, 'test')
