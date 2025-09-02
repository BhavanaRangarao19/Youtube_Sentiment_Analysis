import pandas as pd
import re
from datetime import datetime

def parse_whatsapp(file):
    """
    Parse WhatsApp exported chat text file.
    Returns DataFrame with columns: datetime, author, message
    Ignores system messages where message is empty.
    """
    # with open(file_path, "r", encoding="utf-8") as f:
    #     lines = f.read().splitlines()
    lines = file.read().decode("utf-8").splitlines()
    
    data = []

    # Match both yy and yyyy, author optional (system messages)
    pattern = re.compile(
        r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}\s?[APap][Mm]) - (.*?)(?:: (.*))?$'
    )

    for line in lines:
        line = line.replace('\u202f', ' ').replace('\xa0', ' ').strip()
        match = pattern.match(line)
        if match:
            date_str, time_str, author, message = match.groups()
            message = message if message else ""
            try:
                # Try both yy and yyyy formats
                try:
                    dt = datetime.strptime(f"{date_str}, {time_str}", "%d/%m/%y, %I:%M %p")
                except ValueError:
                    dt = datetime.strptime(f"{date_str}, {time_str}", "%d/%m/%Y, %I:%M %p")
            except ValueError:
                continue
            # Remove edited tag
            message = message.replace("<This message was edited>", "").strip()
            data.append([dt, author, message])
    
    df = pd.DataFrame(data, columns=['datetime', 'author', 'message'])
    
    # Filter out empty messages (system messages)
    df = df[df['message'].str.strip() != ""].reset_index(drop=True)
    return df
