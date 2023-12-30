import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from random import seed, randint
import pandas as pd

from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

import requests
import time


# function that given a company name attempts to return its official facebook page url,
# assuming it is shown as first google-search result using q=company_name+"+Facebook+Page"
# Selenium
def get_facebook_url(company_name):
    driver = webdriver.Chrome()
    try:
        driver.get("https://www.google.com/search?q=" + company_name + "+Facebook+Page")
        # Wait for the search results to load
        time.sleep(5)

        # Locate and click on the official page (assuming it's the first result)
        results = driver.find_element("id", "rso")  # finds webresults
        driver.find_element(
            "xpath", '//*[@id="rso"]/div[1]/div/div/div[1]/div/div/span/a'
        ).click()

        time.sleep(5)

        official_page_url = driver.current_url

        return official_page_url

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

    finally:
        driver.close()
        driver.quit()


# function to get FB followers and likes given company FB url
def get_facebook_followers(fb_url):
    """
    This function uses Selenium to navigate to a Facebook profile and print the number of followers.
    """

    try:
        # Set up the webdriver
        driver = webdriver.Chrome()

        # Navigate to the Facebook profile
        # eg AmkorTechnology
        driver.get(fb_url)
        time.sleep(5)

        # Find num of likes and followers by abs xpath
        try:
            followers_element = driver.find_element(
                "xpath",
                "/html/body/div[1]/div/div[1]/div/div[3]/div/div/div/div[1]/div[1]/div/div/div[1]/div[2]/div/div/div/div[3]/div/div/div[2]/span/a[2]",
            )
            likes_element = driver.find_element(
                "xpath",
                "/html/body/div[1]/div/div[1]/div/div[3]/div/div/div/div[1]/div[1]/div/div/div[1]/div[2]/div/div/div/div[3]/div/div/div[2]/span/a[1]",
            )

            # extract the follower count and likes count
            followers_count = followers_element.text.split()[0]
            likes_count = likes_element.text.split()[0]

        # For pages with only likes and no follower count
        except:
            likes_element = driver.find_element(
                "xpath",
                "/html/body/div[1]/div/div[1]/div/div[3]/div/div/div/div[1]/div[1]/div/div/div[1]/div[2]/div/div/div/div[3]/div/div/div[2]/span/a",
            )

            # Extract the likes count
            likes_count = likes_element.text.split()[0]
            followers_count = 0

        return followers_count, likes_count

    except Exception as e:
        # Log the error
        print(f"Error: {e}")
    finally:
        # Close the webdriver
        driver.close()
        driver.quit()


def text_to_num(text, bad_data_val=0):
    d = {"K": 1000, "M": 1000000, "B": 1000000000}
    if not isinstance(text, str):
        # Non-strings are bad are missing data in poster's submission
        return bad_data_val

    elif text[-1] in d:
        # separate out the K, M, or B
        num, magnitude = text[:-1], text[-1]
        return int(float(num) * d[magnitude])
    else:
        return float(text)


# def get_likes_followers(fb_url):
#     url = str(fb_url) + "friends_likes/"
#     url_likes = str(fb_url) + "friends_likes/"
#     url_followers = str(fb_url) + "followers/"
#     driver = webdriver.Chrome()
#     driver.get(fb_url)
#     get_source = driver.page_source
#     soup = BeautifulSoup(get_source, "html.parser")
#     likes = soup.find("a", {"href": url_likes})
#     followers = soup.find("a", {"href": url_followers})
#     # driver.close()
#     driver.quit()
#     return likes.text, followers.text


# worked briefly


def get_wait(max):
    return randint(0, max)


def get_likes_followers(fb_url):
    likes = 0
    followers = 0

    url = str(fb_url) + "friends_likes/"
    url_likes = str(fb_url) + "friends_likes/"
    url_followers = str(fb_url) + "followers/"

    driver = webdriver.Chrome()
    driver.get(fb_url)
    time.sleep(get_wait(5))
    get_source = driver.page_source
    time.sleep(get_wait(5))
    soup = BeautifulSoup(get_source, "html.parser")
    time.sleep(get_wait(5))
    likes = soup.find("a", {"href": url_likes})
    time.sleep(get_wait(5))
    followers = soup.find("a", {"href": url_followers})
    time.sleep(get_wait(5))
    driver.quit()

    if likes.text:
        likes = likes.text

    if followers.text:
        followers = followers.text

    return likes, followers
    # return likes.text, followers.text
    # return '0', followers.text
