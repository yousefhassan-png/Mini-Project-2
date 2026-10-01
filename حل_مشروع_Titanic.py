# مشروع تنظيف وتحليل بيانات Titanic
# إعداد: طالب واحد

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
# الخطوة 1: تجهيز البيانات الفوضوية
titanic = sns.load_dataset('titanic')

duplicated_rows = titanic.sample(15, random_state=42)
messy_titanic = pd.concat([titanic, duplicated_rows], ignore_index=True)

# توحيد حالة النصوص بشكل متعمد غير متناسق
مؤشرات_الخلط = messy_titanic.sample(20, random_state=1).index
messy_titanic.loc[مؤشرات_الخلط, 'embark_town'] = messy_titanic.loc[مؤشرات_الخلط, 'embark_town'].str.upper()

class_info = pd.DataFrame({
    'pclass': [1, 2, 3],
    'class_description': ['First Class - Luxury', 'Second Class - Standard', 'Third Class - Economy'],
    'deck_access': ['Full Access', 'Limited Access', 'No Access']
})

# الخطوة 2: التحليل الاستكشافي قبل التنظيف
print("\\n" + "=" * 60)
print("الخطوة 2: التحليل الاستكشافي قبل التنظيف")
print("=" * 60)

عدد_الصفوف_قبل, عدد_الأعمدة_قبل = messy_titanic.shape
print(f"الشكل العام: {عدد_الصفوف_قبل} صفًا و{عدد_الأعمدة_قبل} عمودًا.")
print("\\nعدد القيم المفقودة ونسبتها:")
القيم_المفقودة = messy_titanic.isnull().sum()
نسب_الفقد = (القيم_المفقودة / len(messy_titanic) * 100).round(2)
ملخص_الفقد = pd.DataFrame({
    'عدد القيم المفقودة': القيم_المفقودة,
    'نسبة الفقد %': نسب_الفقد
})
print(ملخص_الفقد[ملخص_الفقد['عدد القيم المفقودة'] > 0])

عدد_التكرارات = messy_titanic.duplicated().sum()
print(f"عدد الصفوف المكررة بالكامل: {عدد_التكرارات}")

print("\\nالقيم المختلفة في embark_town قبل التنظيف:")
print(messy_titanic['embark_town'].unique())

print("\\nملخص المشاكل المكتشفة:")
print(
    "تحتوي البيانات على صفوف مكررة أُضيفت عمدًا، وقيم مفقودة في عدة أعمدة، "
    "كما أن قيم embark_town غير متناسقة بسبب اختلاف حالة الأحرف. "
    "هذه المشكلات قد تؤثر في دقة التحليل والرسومات."
)

# الخطوة 3: التنظيف والدمج
print("\\n" + "=" * 60)
print("الخطوة 3: التنظيف والدمج")
print("=" * 60)

# نحذف التكرارات التي أُضيفت عمدًا فقط، مع الحفاظ على الصفوف الأصلية.
عدد_الصفوف_الأصلية = len(titanic)
البيانات_الأصلية = messy_titanic.iloc[:عدد_الصفوف_الأصلية].copy()
الصفوف_المضافة = messy_titanic.iloc[عدد_الصفوف_الأصلية:].copy()
الصفوف_المضافة = الصفوف_المضافة.drop_duplicates()
مفتاح_الأعمدة = list(titanic.columns)
الصفوف_المضافة = الصفوف_المضافة.merge(
    البيانات_الأصلية[مفتاح_الأعمدة].drop_duplicates(),
    on=مفتاح_الأعمدة, how='left', indicator=True
)
الصفوف_المضافة = الصفوف_المضافة.loc[
    الصفوف_المضافة['_merge'] == 'left_only', مفتاح_الأعمدة
]
messy_titanic = pd.concat([البيانات_الأصلية, الصفوف_المضافة], ignore_index=True)
print(f"بعد حذف التكرارات المضافة: {len(messy_titanic)} صفًا.")
assert len(messy_titanic) == عدد_الصفوف_الأصلية, "عدد الصفوف لا يطابق البيانات الأصلية!"

messy_titanic['embark_town'] = messy_titanic['embark_town'].str.title()
print("القيم بعد توحيد embark_town:")
print(messy_titanic['embark_town'].unique())

# الوسيط أقل تأثرًا بالقيم العمرية المتطرفة من المتوسط
وسيط_العمر = messy_titanic['age'].median()
messy_titanic['age'] = messy_titanic['age'].fillna(وسيط_العمر)

# تعويض embarked بالقيمة الأكثر تكرارًا
قيمة_الميناء = messy_titanic['embarked'].mode()[0]
messy_titanic['embarked'] = messy_titanic['embarked'].fillna(قيمة_الميناء)

# حذف deck لأن نسبة الفقد فيه مرتفعة، ما يجعل تعويضه غير موثوق
نسبة_فقد_السطح = messy_titanic['deck'].isna().mean() * 100
messy_titanic = messy_titanic.drop(columns=['deck'])

print(f"وسيط العمر المستخدم: {وسيط_العمر:.2f}")
print(f"قيمة embarked الأكثر تكرارًا: {قيمة_الميناء}")
print(f"نسبة القيم المفقودة في deck قبل الحذف: {نسبة_فقد_السطح:.2f}%")
print("القيم المفقودة بعد المعالجة:")
print(messy_titanic.isnull().sum()[messy_titanic.isnull().sum() > 0])
assert messy_titanic['age'].isnull().sum() == 0
assert messy_titanic['embarked'].isnull().sum() == 0
assert 'deck' not in messy_titanic.columns
# قد تحتوي البيانات الأصلية على صفوف متطابقة؛ نتحقق من عدم بقاء الصفوف المضافة عمدًا.
عدد_التكرارات_المتبقية = messy_titanic.duplicated().sum()
print(f"عدد التكرارات المتبقية من البيانات الأصلية: {عدد_التكرارات_المتبقية}")

# دمج معلومات الفئات
messy_titanic = messy_titanic.merge(class_info, on='pclass', how='left')
assert messy_titanic['class_description'].notna().all()
assert messy_titanic['deck_access'].notna().all()
print(f"بعد الدمج: {messy_titanic.shape[0]} صفًا و{messy_titanic.shape[1]} عمودًا.")
print("تم التحقق من اكتمال معلومات الفئات.")

# الخطوة 4: التقرير البصري
print("\\n" + "=" * 60)
print("الخطوة 4: التقرير البصري")
print("=" * 60)

مجلد_الرسومات = "رسومات_titanic"
os.makedirs(مجلد_الرسومات, exist_ok=True)

# 1. توزيع الركاب حسب الفئة والجنس
plt.figure(figsize=(9, 5))
sns.countplot(data=messy_titanic, x='class_description', hue='sex')
plt.title('Passenger Count by Class and Gender')
plt.xlabel('Passenger Class')
plt.ylabel('Passenger Count')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(مجلد_الرسومات, '01_class_gender.png'), dpi=150)
plt.show()
print("التفسير: يوضح الرسم عدد الركاب في كل فئة مع الفصل حسب الجنس. "
      "يساعد ذلك على مقارنة تركيب الركاب بين الفئات.")

# 2. سعر التذكرة حسب الفئة
plt.figure(figsize=(9, 5))
sns.boxplot(data=messy_titanic, x='class_description', y='fare')
plt.title('Ticket Fare Distribution by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(مجلد_الرسومات, '02_fare_class.png'), dpi=150)
plt.show()
print("التفسير: يعرض الرسم توزيع أسعار التذاكر داخل كل فئة. "
      "يسمح بمقارنة الوسيط ومدى تفاوت الأسعار بين الفئات.")

# 3. نسبة النجاة حسب ميناء الصعود
نسبة_النجاة_الميناء = messy_titanic.groupby('embark_town', observed=False)['survived'].mean().reset_index()
نسبة_النجاة_الميناء['survival_percent'] = نسبة_النجاة_الميناء['survived'] * 100
plt.figure(figsize=(8, 5))
sns.barplot(data=نسبة_النجاة_الميناء, x='embark_town', y='survival_percent')
plt.title('Survival Rate by Embarkation Town')
plt.xlabel('Embarkation Town')
plt.ylabel('Survival Rate (%)')
plt.tight_layout()
plt.savefig(os.path.join(مجلد_الرسومات, '03_survival_port.png'), dpi=150)
plt.show()
print("التفسير: يقارن الرسم متوسط النجاة بين موانئ الصعود. "
      "الاختلافات وصفية ولا تثبت أن الميناء سبب اختلاف النجاة.")

# 4. وجود عائلة وعلاقته بالنجاة — رسم تفاعلي Plotly
messy_titanic['has_family'] = (messy_titanic['sibsp'] + messy_titanic['parch']) > 0
ملخص_العائلة = messy_titanic.groupby('has_family', as_index=False)['survived'].mean()
ملخص_العائلة['حالة_العائلة'] = ملخص_العائلة['has_family'].map({True: 'With family', False: 'Alone'})
ملخص_العائلة['survival_percent'] = ملخص_العائلة['survived'] * 100
شكل_تفاعلي = px.bar(
    ملخص_العائلة, x='حالة_العائلة', y='survival_percent',
    title='Survival Rate: Travelling with Family vs Alone',
    labels={'حالة_العائلة': 'Travel Group', 'survival_percent': 'Survival Rate (%)'},
    text_auto='.1f'
)
شكل_تفاعلي.write_html('titanic_interactive.html', include_plotlyjs=True)
شكل_تفاعلي.show()
print("التفسير: يقارن الرسم نسبة النجاة بين من سافروا مع أفراد عائلة ومن سافروا دونهم. "
      "يعرض ارتباطًا وصفيًا ولا يثبت علاقة سببية.")

# 5. متوسط العمر حسب الفئة وحالة النجاة
messy_titanic['survival_label'] = messy_titanic['survived'].map({0: 'Did not survive', 1: 'Survived'})
plt.figure(figsize=(10, 5))
sns.barplot(data=messy_titanic, x='class_description', y='age', hue='survival_label', errorbar=None)
plt.title('Average Age by Class and Survival Status')
plt.xlabel('Passenger Class')
plt.ylabel('Average Age')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(مجلد_الرسومات, '05_age_class_survival.png'), dpi=150)
plt.show()
print("التفسير: يقارن الرسم متوسط العمر بين الفئات مع الفصل حسب حالة النجاة. "
      "يساعد على ملاحظة الفروق العمرية دون اعتبارها تفسيرًا منفردًا للنجاة.")

# الخطوة 5: التوثيق النهائي
print("\\n" + "=" * 60)
print("الخطوة 5: التوثيق النهائي")
print("=" * 60)
print("قرارات التنظيف:")
print("- حذف الصفوف المكررة باستخدام drop_duplicates.")
print("- توحيد embark_town باستخدام str.title.")
print("- تعويض age بالوسيط لأنه أقل تأثرًا بالقيم المتطرفة.")
print("- تعويض embarked بالمنوال لأنه متغير فئوي.")
print(f"- حذف deck بسبب ارتفاع نسبة الفقد ({نسبة_فقد_السطح:.2f}%).")
print("- دمج class_info باستخدام left merge على pclass.")
print(f"قبل التنظيف: {عدد_الصفوف_قبل} صفًا و{عدد_الأعمدة_قبل} عمودًا.")
print(f"بعد التنظيف والدمج: {messy_titanic.shape[0]} صفًا و{messy_titanic.shape[1]} عمودًا.")

print("\\nالملخص التنفيذي:")
print(
    "يعرض التقرير اختلافًا في أعداد الركاب بين درجات السفر وبين الجنسين، "
    "ما يوفر صورة عن تركيبة المسافرين على متن السفينة. كما يبيّن توزيع الأسعار "
    "فروقًا بين درجات السفر، وهو ما يساعد على فهم اختلاف مستوى التذاكر."
)
print(
    "وتوضح الرسومات كذلك اختلاف نسب النجاة بحسب ميناء الصعود وبحسب السفر مع العائلة، "
    "إضافة إلى مقارنة الأعمار بين الفئات وحالات النجاة. هذه النتائج وصفية وتساعد على "
    "طرح أسئلة إضافية، ولا تكفي وحدها لإثبات أسباب النجاة."
)

messy_titanic.to_csv('titanic_cleaned.csv', index=False)
print("\\nاكتمل التنفيذ. تم حفظ الرسومات داخل مجلد رسومات_titanic، "
      "والرسم التفاعلي في titanic_interactive.html، والبيانات النظيفة في titanic_cleaned.csv.")
