from parsers.whatsapp_parser import parse_whatsapp
df_whatsapp = parse_whatsapp(r'D:\Projects\sampleChat.txt')
print("First 5 lines of parsed DataFrame:")
print(df_whatsapp.head())
print(f"\nTotal messages parsed: {len(df_whatsapp)}")