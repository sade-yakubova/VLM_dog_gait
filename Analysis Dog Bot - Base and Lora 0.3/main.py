from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    error
)
from telegram.ext import (
    CallbackQueryHandler,
    ConversationHandler,
    InvalidCallbackData,
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)
from tocluster import *

from functools import wraps
import json
import logging, sys
import html
import traceback
from telegram.constants import ParseMode
import datetime
import os
# logger = logging.getLogger(__name__)
# from dotenv import load_dotenv
# load_dotenv()

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

from warnings import filterwarnings
from telegram.warnings import PTBUserWarning
filterwarnings(action="ignore", message=r".*CallbackQueryHandler", category=PTBUserWarning)

# os.environ["TOKEN"]

from constants import *

START, STARTCHAT, CANCEL, WAIT_VIDEO, WAIT_AGE, WAIT_BREAD, WAIT_NOTE = range(0, 7)
PROMPT = "You helpful assistant."
MESSAGE = """Describe the dog's gait from the video, paying attention to:
1. Symmetry of limb movements.
2. The load on the front/rear legs.
3. Signs of lameness or muscle atrophy.
4. Pitch characteristics (length, rhythm, tail position).
Find the dog's musculoskeletal problems.

Try looking up the name of the disease. But only if you are sure that the dog has that particular disease. If you are not sure, don't give the exact diagnosis."""

os.makedirs("sessions", exist_ok=True)

# Обработчик ошибок
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
    tb_string = "".join(tb_list)
    update_str = update.to_dict() if isinstance(update, Update) else str(update)
    message = (
        "An error occurred while processing 'update': \n"
        f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
        "</pre>\n\n"
        f"<pre>context.chat_data = {html.escape(str(context.chat_data))}</pre>\n\n"
        f"<pre>context.user_data = {html.escape(str(context.user_data))}</pre>\n\n"
        f"<pre>context.user_data = {html.escape(str(context.bot_data))}</pre>\n\n"
        f"<pre>{html.escape(tb_string)}</pre>"
    )

    await context.bot.send_message(
        chat_id=DEVELOPER_CHAT_ID, text=message, parse_mode=ParseMode.HTML
    )

# Функция доступа. Вход разрешен только ADMINS
def restricted(func):
    @wraps(func)
    async def wrapped(update: Update, context, *args, **kwargs):
        user_id = update.effective_user.id
        if user_id not in ADMINS:
            text = "Sorry, access to the bot has been denied."
            await update.effective_message.reply_text(text=text)
            logging.info(f"Unauthorized access is denied for {user_id}.")
            return
        return await func(update, context, *args, **kwargs)
    return wrapped

@restricted # включение обработки доступа для start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Текст:
    # text = "An AI assistant based on a multimodal visual-linguistic model capable of analyzing a dog's gait and detecting musculoskeletal diseases."
    text = "Hi! I’m GaitMate, your virtual vet assistant. Let’s take care of your dog together."
    # Кнопка. 
    keyboard = [[InlineKeyboardButton(text="Start", callback_data=str(STARTCHAT))]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    video = 'dog-run.mp4'
    if update.callback_query: # при нажатии на кнопку back to start: 
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(text=text, reply_markup=reply_markup)
    else: # при вводе команды /start:
        # await update.message.reply_text(text=text, reply_markup=reply_markup)
        await update.message.reply_video(video=video,
                                         caption=text,
                                         reply_markup=reply_markup)

async def start_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = """Please upload a short video of your dog walking from the side.
To ensure accurate analysis, follow these guidelines:
    1. Camera angle:
        Record from the side view, keeping the dog’s entire body in the frame at all times (head to tail, all legs visible).
    2. Walking path:
        Make your dog walk in a straight line on a flat surface (like a pavement or a hallway).
        Avoid turns, obstacles, or uneven ground.
    3. Distance and framing:
        Stand at a distance where the entire dog fits clearly in the frame (avoid close-ups).
        Keep the camera steady — avoid shaking or following the dog too closely.
    4. Lighting:
        Record in daylight or well-lit conditions so the dog’s movements and limbs are clearly visible.
    5. Duration:
        Aim for 5 to 10 seconds of continuous walking."""
    keyboard = [[InlineKeyboardButton(text="Cancel", callback_data=str(CANCEL))]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.answer()
    # await update.effective_message.edit_text(text=text, reply_markup=reply_markup)
    await update.effective_user.send_message(text=text, reply_markup=reply_markup)
    return WAIT_VIDEO

async def age(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # user_data это всегда папка в которую мы сохраняем данные пользователя
    user_data = context.user_data
    # проверяем gif или video
    file = update.message.animation or update.message.video
    # получаем id пользователя который отправил видео или gif
    user_id = update.effective_user.id
    ###
    keyboard = [[InlineKeyboardButton(text="Cancel", callback_data=str(CANCEL))]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    # получаем время now 
    timenow = datetime.datetime.now()
    timenow = datetime.datetime.strftime(timenow, "%y%m%d-%H%M%S")
    ###
    user_data["session"] = timenow
    # создать папку sessions/user_data["session"] если ее нет
    os.makedirs("sessions/" + user_data["session"], exist_ok=True)
    # получаем файл который был отправлен
    try: 
        getfile = await file.get_file()
    except error.BadRequest as err:
        if "File is too big" in err:
            text = "The file is too big, try another one."
        else:
            text = "Problems with load video."
        await update.effective_message.reply_text(
                        text=text,
                        reply_markup=reply_markup)
        return WAIT_VIDEO
    # Сохраняем файл
    file_name = file.file_name
    file_path = rf'sessions/{user_data["session"]}/{file_name}'
    await getfile.download_to_drive(file_path)
    # Сохраняем файл в папку пользователя
    user_data['video_path'] = file_path
    #
    text = "How old is your dog?\nIf you have difficulty, write - unknown."
    await update.effective_message.reply_text(text=text, reply_markup=reply_markup)
    return WAIT_AGE

async def bread(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message.text
    context.user_data["age"] = message
    text = "What breed is your dog? (If mixed, describe the main traits, size)\nIf you have difficulty, write - unknown."
    keyboard = [[InlineKeyboardButton(text="Cancel", callback_data=str(CANCEL))]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.effective_message.reply_text(text=text, reply_markup=reply_markup)
    return WAIT_BREAD

async def note(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message.text
    context.user_data["bread"] = message
    text = "Have you noticed any signs of discomfort — like limping, stiffness, tiredness, or avoiding one leg? If yes, could you tell me more? (Which leg, when it started, how it affects activity, etc.)\nIf you have difficulty, write - unknown."
    keyboard = [[InlineKeyboardButton(text="Cancel", callback_data=str(CANCEL))]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.effective_message.reply_text(text=text, reply_markup=reply_markup)
    return WAIT_NOTE

async def final(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_data = context.user_data
    user_data["prompt"] = PROMPT # теперь промпт стабильно всегда одинаков и един
    user_data["text"] = f"""The dog in the video is:
Age: {user_data.get("age")}
Breed: {user_data.get("bread")}
Note: {update.effective_message.text}

{MESSAGE}
"""
    session = user_data["session"]
    data_path = user_data["data_path"] = f"sessions/{session}/data.json"
    user_data['user_id'] = update.effective_user.id
    # Сохраняем данные user_data в файл data.json
    # (в дальнейшем data.json данные считываются в analysis.py)
    with open(data_path, 'w', encoding='utf-8') as file:
        data = json.dumps(user_data, ensure_ascii=False)
        file.write(data)
    # Отправляем сообщение пользователю:
    text = f"It may take 5 to 30 minutes for the results of request #{session} to arrive" 
    await update.effective_message.reply_text(text=text)
    # Запускаем моментально функция waiting_process
    context.job_queue.run_once(waiting_process, 0, user_id=update.effective_user.id)
    # Выход из цепочки ConversationHandelr:
    return ConversationHandler.END

async def waiting_process(context: ContextTypes.DEFAULT_TYPE) -> None:
    print('waiting process')
    user_id = context.user_data['user_id']
    session = context.user_data['session']
    # Запускаем функцию process, которая формирует задачу и отправляет ее в кластер
    result = await process(context.user_data)
    # в переменной result хранится конечный результат анализа, если не было никаких ошибок
    # text = f"Response to request #{session}:\n{result}"
    text = f"""Response to request #{session}:
    {result}


    ⚠️ Please note: I am a virtual assistant and not a licensed veterinarian.
    The information I provide is for informational purposes only and is not a substitute for professional veterinary diagnosis or treatment.
    If your dog is in pain or symptoms worsen, please consult a licensed vet as soon as possible."""
    if result:
        await context.bot.send_message(chat_id=user_id, text=text)
    else:
        text = f"An error occurred on request #{session}!"
        await context.bot.send_message(chat_id=user_id, text=text)

async def handle_invalid_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.callback_query.answer()
    event = update.effective_message.to_dict()
    photo = event.get("photo", None)
    video = event.get("video", None)
    text = "Sorry, the button is out of date 😕 Please re-enter the /start command"
    if photo or video:
        await update.effective_user.send_message(text=text)
    else:
        await update.effective_message.edit_text(text=text)

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    keyboard = [[InlineKeyboardButton(text="Back to start", callback_data=str(START))]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.answer()
    await query.edit_message_text('Cancel.', reply_markup=reply_markup)
    return ConversationHandler.END

def main() -> None:
    # Build app
    app = ApplicationBuilder().token(TOKEN).arbitrary_callback_data(True).build()
    conv_chat = ConversationHandler(
        # при нажатии на кнопку start мы входим в цепочку ConversationHandler
        entry_points=[CallbackQueryHandler(start_chat, pattern=f"^{STARTCHAT}$")],
        states={
            WAIT_VIDEO: [MessageHandler(filters.VIDEO | filters.ANIMATION, age)],
            WAIT_AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, bread)],
            WAIT_BREAD: [MessageHandler(filters.TEXT & ~filters.COMMAND, note)],
            WAIT_NOTE: [MessageHandler(filters.TEXT & ~filters.COMMAND, final)],
        },
        # fallbacks - выйти из цепочки:
        fallbacks=[CallbackQueryHandler(cancel, pattern=f"^{CANCEL}$"),
                   CommandHandler('start', start)]
    )
    # Регистрация функции error_handler, которая запускается при ошибке
    app.add_error_handler(error_handler)
    # Регистрируем conv_chat (ConversationHandler) - та самая цепочка
    app.add_handler(conv_chat)
    # message command /start
    app.add_handler(CommandHandler('start', start))
    # Регистрируем кнопку Back to start
    app.add_handler(CallbackQueryHandler(start, pattern=f"^{START}$"))
    # Если после перезапуска нажать на кнопку, то выйдет ошибка, за это отвечает следующее:
    app.add_handler(
        CallbackQueryHandler(handle_invalid_button, pattern=InvalidCallbackData)
    )
    # Запуск бота
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()