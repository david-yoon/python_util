import re
from bs4 import BeautifulSoup
import unicodedata

SUPPORTING_CONFIG = ["strip_html_tag", "remove_url", "remove_non_ascii", "helpx_specific"]

class TextPreprocess():
    
    def __init__(self, process_config):
        
        print("process_config: ", process_config)
        self.set_strip_html_tag = False
        self.set_remove_url = False
        self.set_remove_non_ascii = False
        self.set_helpx_specific = False
        
        for config in process_config:
            
            if config not in SUPPORTING_CONFIG:
                print("[ERROR] supporting config: ", SUPPORTING_CONFIG)
                return
            
            if config == "strip_html_tag":
                self.set_strip_html_tag = True
            
            if config == "remove_url":
                self.set_remove_url = True
                
            if config == "remove_non_ascii":
                self.set_remove_non_ascii = True
                
            if config == "helpx_specific":
                self.set_helpx_specific = True

        
    def _strip_html_tags(self, text):
        try:
            soup = BeautifulSoup(text, "html.parser")
            ret = soup.get_text()
        except:
            print(text)

        return ret 
        
    def _remove_url(self, text):
        return re.sub(r"http\S+", "", text)

    
    def _remove_non_ascii(self, words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words.split():
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
            
        ret = " ".join(new_words)
            
        return ret
    
    
    def _helpx_specific(self, text):
        
        # remove repeated dot
        text = text.replace("..", "")
        text = text.replace(". .", "")
        
        # remove date 00-00-00
        pattern = r"\d{4}-\d{2}-\d{2}"
        text = re.sub(pattern, "", text)
        
        # remove time 00:00:00
        pattern = r"\d{2}:\d{2}:\d{2}"
        text = re.sub(pattern, "", text)
        
        # remove library package
        if ("com." in text) or ("org." in text):
            # skip this sentence
            text = ""
        
        return text
    
    
    def run(self, text):
        if self.set_strip_html_tag:
            text = self._strip_html_tags(text)

        if self.set_remove_url:
            text = self._remove_url(text)
            
        if self.set_remove_non_ascii:
            text = self._remove_non_ascii(text)            
            
        if self.set_helpx_specific:
            text = self._helpx_specific(text)
            
        text_remove_duplicate_space = " ".join(text.split())
        
        return text_remove_duplicate_space
    
    
def test():

    test_text = "<html>hello</html> http://me.io"
    process = TextPreprocess(["strip_html_tag", "remove_url", "remove_non_ascii", "helpx_specific"])
    
    print("before: ", test_text)
    print("after: ", process.run(test_text))
    
    
if __name__ == "__main__":
    
    test()
    