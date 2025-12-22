if __name__ == "__main__":
    # 2301：光寶科
    df_2301 = fetch_prepare_recalc("2301.TW")
    save_to_firestore(df_2301, "2301.TW")

    # ➕ 2408：南亞科（同方法）
    df_2408 = fetch_prepare_recalc("2408.TW")
    save_to_firestore(df_2408, "2408.TW")

    # ➕ 加權指數 / 電子指數
    save_index_close("^TWII", "TAIEX")
    save_index_close("^TWTE", "ELECTRONICS")

    print("2301 tail:\n", df_2301.tail())
    print("2408 tail:\n", df_2408.tail())
