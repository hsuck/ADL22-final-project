from typing import *
from pathlib import Path
from datetime import datetime
from datasets import Dataset
try:
    from .interface import Preprocessor
    from .vocab import Vocab
except:
    from interface import Preprocessor
    from vocab import Vocab

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
    def __init__(self, column_names):
        self.column_names = column_names
        self.datetime_format = "%Y-%m-%d %H:%M:%S"
        
        self.__encode_func__ = [
            self.encode_course_id,
            self.encode_sentences, # course name
            self.encode_course_price,

            self.encode_teacher_id,
            self.encode_sentences, # teacher intro

            self.encode_groups,
            self.encode_subgroups,
            self.encode_topics,

            self.encode_published_time, 

            self.encode_sentences, # course desc
            self.encode_sentences, # will learn
            self.encode_sentences, # require tool
            self.encode_sentences, # background
            self.encode_sentences, # target group
        ]
        self.__none_value__ = [ Vocab.UNK ] * len(self.__encode_func__)

    @staticmethod
    def interest_generator(interests: str):
        for interest in interests.split(','):
            group, subgroup = interest.split('_')
            yield (group, subgroup)

    def fill_none_as_unk(self,  batch: Dict[str, List[str]]) -> Dict[str, List[str]]:
        for column, value in zip(self.column_names, self.__none_value__):
            batch[column] = [
                data if data != None else value
                for data in batch[column]
            ]
        return batch


    def encode(self, course_profile: Dict[str, List[str]]) -> Dict[str, List[ Union[int, List[int]] ]]:
        res = {
            key: func( course_profile[key] ) for key, func in zip(self.column_names, self.__encode_func__)
        }
        return res
    
    def encode_sentences(self, sents: List[str]) -> List[ List[int] ]:
        print("Encode Sent")
        raise NotImplementedError

    def encode_course_id(self, course_id: List[str]) -> List[ int ]:
        raise NotImplementedError

    def encode_course_price(self, course_price: List[int]) -> List[int]:
        return course_price
    
    def encode_teacher_id(self, teacher_id: List[str]) -> List[int]:
        raise NotImplementedError

    def encode_groups(self, groups: List[str]) -> List[ List[int] ]:
        raise NotImplementedError
    def encode_subgroups(self, sub_groups: List[str]) -> List[ List[int] ]:
        raise NotImplementedError
    def encode_topics(self, topics: List[str]) -> List[ List[int] ]:
        raise NotImplementedError

    def encode_published_time(self, published_time: List[str]) -> List[datetime]:
        " ex: 2015-05-20 19:56:05 or 2015-05-20 19:56:05.158 "
        max_len = len("YYYY-mm-dd HH:MM:SS")
        return [
            datetime.strptime(t[:max_len], self.datetime_format) for t in published_time
        ]
    

class BasicCoursePreprocessor(CoursePreprocessor):

    def __init__(self, vocab_dir, column_names, year_offset=2014, month_offset=12):
        super().__init__(column_names)

        self.vocab_dir = Path(vocab_dir)

        self.files = [
            'course.json',
            'teacher.json',
            'group.json',
            'subgroup.json',
            'topic.json',
        ]

        self.encoder: Dict[str, Vocab] = {}
        
        for file in self.files:
            file_path: Path = self.vocab_dir / file
            assert file_path.exists(), f"{file_path} doesn't exist"
            vocab = Vocab()
            vocab.load( file_path )
            self.encoder[ file_path.stem ] = vocab
        
        # For course with unknown publish time, set it as "min time"
        self.offset = (year_offset, month_offset)
        self.__none_value__[8] = f"{year_offset}-{month_offset}-01 00:00:00.000"

    def encode(self, course_profile: Dict[str, List[str]]) -> Dict[str, List[Union[int, List[int]]]]:
        # skip all sentence property
        res = {
            key: func( course_profile[key] )
            for key, func in zip(self.column_names, self.__encode_func__)
            if func != self.encode_sentences
        }
        return res


    def encode_csv_sent(self, csv_sents: List[str], encoder: Vocab) -> List[ List[int] ]:
        return [ encoder.encode(x.split(',')) for x in csv_sents ]

    def encode_tokens(self, tokens: List[str], encoder: Vocab) -> List[ int ]:
        return encoder.encode(tokens)

    def encode_course_id(self, course_id: List[str]) -> List[int]:
        return self.encode_tokens(course_id, self.encoder['course'])

    def encode_course_price(self, course_price: List[int]) -> List[int]:
        return super().encode_course_price(course_price)
    
    def encode_teacher_id(self, teacher_id: List[str]) -> List[int]:
        return self.encode_tokens(teacher_id, self.encoder['teacher'])

    def encode_groups(self, groups: List[str]) -> List[List[int]]:
        return self.encode_csv_sent(groups, self.encoder['group'])

    def encode_subgroups(self, sub_groups: List[str]) -> List[List[int]]:
        return self.encode_csv_sent(sub_groups, self.encoder['subgroup'])

    def encode_topics(self, topics: List[str]) -> List[List[int]]:
        return self.encode_csv_sent(topics, self.encoder['topic'])
    
    def encode_published_time(self, published_time: List[str]) -> List[int]:
        time = super().encode_published_time(published_time)
        res = [
            (t.year - self.offset[0])*12 + (t.month - self.offset[1])for t in time
        ]
        return res

def prepare_course_datasets( course_data: Dataset, course_p: CoursePreprocessor, batch_size: int, with_bos_eos: bool ) -> Dataset:
    D = course_data.map(
        course_p.fill_none_as_unk,
        batch_size=batch_size,
        batched=True,
    )
    D = D.map(
        course_p.encode,
        batch_size=batch_size,
        batched=True,
    )
    # D = D.sort("course_id")
    return D

if __name__ == "__main__":
    course_data = Dataset.from_csv( "../../data/courses.csv" )
    course_p = BasicCoursePreprocessor("../../cache/vocab", column_names=course_data.column_names)
    batch_size = 32

    X = course_data.select( range(5) )
    Y = prepare_course_datasets( X, course_p, batch_size, False )
    
    # print( X['course_published_at_local'] )
    # print()
    # print(X.column_names)
    for column in X.column_names:
        if column in ['course_name','teacher_intro', 'description', 'will_learn', 'required_tools', 'recommended_background', 'target_group']: continue
        print(f"[{column}]:")
        print(f"    {X[column]}")
        print(f" -> {Y[column]}")


