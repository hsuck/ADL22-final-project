from interface import Preprocessor
class CoursePreprocessor(Preprocessor):
    """
        - course_id: hash str
        - course_name: str
        - course_price: int

        - teacher_id: hash str
        - teacher_intro: sentences

        - groups: 課程分類,逗號分隔多項. ex: '程式,行銷','程式,設計','職場技能'
        - sub_groups: 課程子分類，使用逗號分隔多項. ex: "更多語言,歐洲語言"
        - topics: 課程主題,逗號分隔多項. 'PowerPoint,簡報技巧'

        - course_published_at_local: datetime "YYYY-mm-dd H-M-S"

        - description: markdown / html ?
        - will_learn: sentences
        - required_tools: sentences
        - recommended_background: sentences
        - target_group: sentences
    """

    def encode_course_id(self, course_id):
        raise NotImplementedError
    def encode_course_name(self, course_name):
        raise NotImplementedError
    def encode_course_price(self, course_price):
        raise NotImplementedError
    
    def encode_teacher_id(self, teacher_id):
        raise NotImplementedError
    def encode_teacher_intro(self, teacher_intro):
        raise NotImplementedError

    def encode_groups(self, groups):
        raise NotImplementedError
    def encode_sub_groups(self, sub_groups):
        raise NotImplementedError
    def encode_topics(self, topics):
        raise NotImplementedError

    def encode_published_time(self, published_time):
        raise NotImplementedError
    
    def encode_description(self, description):
        raise NotImplementedError
    def encode_will_learn(self, will_learn):
        raise NotImplementedError
    def encode_required_tools(self, required_tools):
        raise NotImplementedError
    def encode_target_group(self, target_group):
        raise NotImplementedError
