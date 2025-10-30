import os
import re

def name_file(book_name, chap_num, year):
    return book_name + " " + str(chap_num) + " " + str(year) + ".txt"

def text_to_chapter_list(book):
    chapters_list = []
    
    if re.search(r'\n\s*[IVX]+\.\s+', book):
        chapters_list = re.split(r'\n\s*([IVX]+\.)\s+', book)
        cleaned_chapters = []
        for i in range(1, len(chapters_list), 2):
            if i + 1 < len(chapters_list):
                chapter = chapters_list[i] + " " + chapters_list[i + 1]
                cleaned_chapters.append(chapter.strip())
        chapters_list = cleaned_chapters
    
    elif re.search(r'\n\s*\d+\.\s*\n', book):
        chapters_list = re.split(r'\n\s*(\d+\.)\s*\n', book)
        cleaned_chapters = []
        for i in range(1, len(chapters_list), 2):
            if i + 1 < len(chapters_list):
                chapter = chapters_list[i] + "\n" + chapters_list[i + 1]
                cleaned_chapters.append(chapter.strip())
        chapters_list = cleaned_chapters
    
    # Pattern 3: Plain numbered sections (Human All Too Human)
    elif re.search(r'\n\s*\d+\s*\n', book):
        chapters_list = re.split(r'\n\s*(\d+)\s*\n', book)
        # Remove empty strings and combine section numbers with content
        cleaned_chapters = []
        for i in range(1, len(chapters_list), 2):
            if i + 1 < len(chapters_list):
                chapter = chapters_list[i] + "\n" + chapters_list[i + 1]
                cleaned_chapters.append(chapter.strip())
        chapters_list = cleaned_chapters
    
    # Fallback: Traditional chapter splitting
    else:
        chapters_list = book.split("CHAPTER ")
    
    # Remove empty chapters
    chapters_list = [chapter for chapter in chapters_list if chapter.strip()]
    
    return chapters_list


def get_book_list():
    folder = r"C:\Users\User\Desktop\NLPMini\BackUp\Nietche\Rawtxt"
    books_names = get_books_names(folder)

    books_list = []
    for book_name in books_names:
        year = get_year(book_name)
        print("the book name is: " + book_name + " and the year is: " + year)

        book_path = folder + "\\" + book_name + ".txt"

        with open(book_path, "r", encoding="utf-8") as file:
            book_text = file.read()
        books_list.append({'text': book_text, 'year': year})
    return books_list


def get_chap_list(path):
    folder = r"C:\Users\User\Desktop\NLPMini\BackUp\Nietche\Rawtxt"
    books_names = get_books_names(folder)
    chap_list = []
    for book_name in books_names:

        chap_list = []
        year = get_year(book_name)
        print("the book name is: " + book_name + " and the year is: " + year)

        book_path = folder + "\\" + book_name + ".txt"

        with open(book_path, "r", encoding="utf-8") as file:
            book_text = file.read()

        clean_book_text = clean_book(book_text)

        chapters = text_to_chapter_list(clean_book_text)

        for chapter in chapters:
            # create a file for each chapter
            chap_list.append([chapter, year])
    return chap_list


def clean_book(book_text):
    start = find_position_chap1(book_text) + 8
    end = find_position_last_chap(book_text)
    clean_book_text = book_text[start:end]
    return clean_book_text


def get_year(book_filename):
    # Hardcoded mapping of filename to publication year
    years = {
        '1The Birth Of Tragedy.txt': '1872',
        '1Untimely Meditations.txt': '1873',
        '2Human All Too Human.txt': '1878',
        '2The Dawn of Day.txt': '1881',
        '2The Gay Science.txt': '1882',
        '3Thus Spake Zarathustra.txt': '1883',
        '3Beyond Good And Evil.txt': '1886',
        '4The Antichrist.txt': '1895',
        '4The Twilight of the Idols.txt': '1888',
    }
    return years.get(book_filename, 'Unknown')


def extract_untimely_meditations(full_text):
    # Extract only the 'Untimely Meditations' (Thoughts Out of Season) section
    start_marker = 'THOUGHTS OUT OF SEASON'
    end_markers = [
        'WE PHILOLOGISTS',
        'THE ANTICHRIST',
        'THE DAWN OF DAY',
        'THE BIRTH OF TRAGEDY',
        'EARLY GREEK PHILOSOPHY',
        'ON THE FUTURE OF OUR',
        'HUMAN',
        'THE JOYFUL WISDOM',
        'THE CASE OF WAGNER',
        'ECCE HOMO',
        'THE TWILIGHT OF THE IDOLS',
        'THE GENEALOGY OF MORALS',
        'THE WILL TO POWER',
    ]
    start_idx = full_text.find(start_marker)
    if start_idx == -1:
        return ''
    # Find the first end marker after the start
    end_idx = len(full_text)
    for marker in end_markers:
        idx = full_text.find(marker, start_idx + len(start_marker))
        if idx != -1 and idx < end_idx:
            end_idx = idx
    return full_text[start_idx:end_idx].strip()


def get_books_names(folder):
    books_names = []
    for file in os.listdir(folder):
        name, ext = os.path.splitext(file)
        if ext.lower() == '.txt':
            # remove digits at the start & whitespaces
            clean = re.sub(r'^\d+', '', name).strip()
            books_names.append(clean)
    return books_names


def find_position_chap1(text):
    position = text.find('*** START OF THE PROJECT GUTENBERG EBOOK')
    start_search = text[position:]
    
    patterns = [
        # For Thus Spake Zarathustra - look for FIRST PART or I.
        r'FIRST PART',
        r'\n\s*I\.\s+',
        # For numbered sections
        r'\n\s*1\.\s*\n',
        r'\n\s*2\.\s*\n',
        # For preface/introduction end patterns
        r'\n\s*1\s*\n',
        # Traditional chapter patterns
        r'CHAPTER I\.',
        r'Chapter I\.',
        r'CHAPTER 1\.'
    ]
        
    for pattern in patterns:
        match = re.search(pattern, start_search)
        if match:
            return position + match.start()

    # Fallback patterns
    position = text.find("(start)")
    if position == -1:
        # Look for numbered sections at the start
        match = re.search(r'\n\s*1\.\s*\n', text)
        if match:
            position = match.start()
    if position == -1:
        # Look for Roman numerals
        match = re.search(r'\n\s*I\.\s+', text)
        if match:
            position = match.start()
    
    return position


def find_position_last_chap(text):
    position = text.find('*** END OF THE PROJECT GUTENBERG EBOOK')
    return position


def main():
    folder = r"C:\Users\User\Desktop\NLPMini\BackUp\Nietche\Rawtxt"
    books_names = os.listdir(folder)
    chaps_path = r"C:\Users\User\Desktop\NLPMini\BackUp\Nietche\CleanChapters"

    for book_filename in books_names:

        if not book_filename.endswith('.txt'):
            continue
        year = get_year(book_filename)
        book_name = re.sub(r'^\d+', '', book_filename[:-4]).strip()
        print("the book name is: " + book_name + " and the year is: " + year)

        book_path = os.path.join(folder, book_filename)
        with open(book_path, "r", encoding="utf-8") as file:
            book_text = file.read()

        # Special handling for Untimely Meditations
        if book_filename == '1Untimely Meditations.txt':
            book_text = extract_untimely_meditations(book_text)

        clean_book_text = clean_book(book_text)
        chapters = text_to_chapter_list(clean_book_text)

        for chap_num, chapter in enumerate(chapters):
            chaps_path = r"C:\Users\User\Desktop\NLPMini\BackUp\Nietche\CleanChapters"
            chap_path = os.path.join(chaps_path, f"{book_name} ({year}) {chap_num}.txt")
            with open(chap_path, "w", encoding="utf-8") as file:
                file.write(chapter)

    # counting number of chapters
    count = 0
    for file in os.listdir(chaps_path):
        if file.endswith(".txt"):
            count += 1

    print("number of chapters is: " + str(count))
    return


if __name__ == "__main__":
    main()
