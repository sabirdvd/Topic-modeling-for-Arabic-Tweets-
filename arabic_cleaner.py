import re
import pandas
import sys
from tqdm import tqdm
import pyarabic.araby as araby

def arabic_only(csv_file) -> pandas.DataFrame:
    print(f'Remove all but Arabic; source file {csv_file}')
    csv = pandas.read_csv(csv_file)
    l = []
    for i in tqdm(csv['text'].iloc[0:]):
        results = re.sub(r'[^0-9\u0600-\u06ff\u0750-\u077f\ufb50-\ufbc1\ufbd3-\ufd3f\ufd50-\ufd8f\ufd50-\ufd8f\ufe70-\ufefc\uFDF0-\uFDFD]+', ' ', i)
        l.append(results)
        df = pandas.DataFrame(l, columns=['text'])

    #df.to_csv(f'cleaned_{csv_file}', index=False)
    print(f'Finished removing non-arabic...starting normalization')
    normalizeArabic(df)

def normalizeArabic(DataFrame) -> araby:
    l= []
    print(f'Started Normalization')
    for i in tqdm(DataFrame['text'].iloc[0:]):
        #results = re.sub(r'[^0-9\u0600-\u06ff\u0750-\u077f\ufb50-\ufbc1\ufbd3-\ufd3f\ufd50-\ufd8f\ufd50-\ufd8f\ufe70-\ufefc\uFDF0-\uFDFD]+', ' ', i)
        #df = pandas.DataFrame(l, columns=['text'])

    	i = i.strip()
    	i = re.sub("[إأٱآا]", "ا", i)
    	i = re.sub("ى", "ي", i)
    	i = re.sub("ؤ", "ء", i)
    	i = re.sub("ئ", "ء", i)
    	i = re.sub("ة", "ه", i)
    	noise = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    	i = re.sub(noise, '', i)
    	i = re.sub(r'(.)\1+', r"\1\1", i) # Remove longation
    	l.append(araby.strip_tashkeel(i))
    	df = pandas.DataFrame(l, columns=['text'])

    df.to_csv(f'cleaned_{csv_file}', index=False)
  # return araby.strip_tashkeel(i)

if __name__ == '__main__':
    csv_file = sys.argv[1]
    print(sys.argv)
   # if len(sys.argv) > 2:
   #     limit = sys.argv[2]
   # else:
   #     limit = None
    arabic_only(csv_file)
