import pandas as pd
import scipy.stats as stats


class CategoricalFeature:
    def __init__(
        self,
        df: pd.DataFrame,
        feature: str,
        label: str,
        missing: any = 'MISSING'
    ) -> None:
        self.df = df
        self.feature = feature
        self.label = label
        self.missing = missing

    @property
    def df_lite(self) -> pd.DataFrame:
        df_lite = self.df
        df_lite['bin'] = df_lite[self.feature].fillna(self.missing)
        return df_lite[['bin', self.label]]


class ContinuousFeature:
    def __init__(
        self,
        df: pd.DataFrame,
        feature: str,
        label: str,
        bin_min_size: float = 0.05,
        missing: any = 'MISSING'
    ) -> None:
        self.df = df
        self.feature = feature
        self.label = label
        self.missing = missing
        self.bin_min_size = int(len(self.df) * bin_min_size)

    def __generate_bins__(self, bins_num: int) -> pd.DataFrame:
        df = self.df[[self.feature, self.label]]
        df['bin'] = pd.qcut(df[self.feature], bins_num, duplicates='drop') \
            .apply(lambda x: x.left) \
            .astype(float)
        return df

    def __generate_correct_bins__(self, bins_max: int = 20) -> pd.DataFrame:
        for bins_num in range(bins_max, 1, -1):
            df = self.__generate_bins__(bins_num)
            df_grouped = df.groupby('bin') \
                .agg({self.feature: 'count', self.label: 'sum'})
            df_grouped = pd.DataFrame(df_grouped).reset_index()
            r, p = stats.stats.spearmanr(
                df_grouped['bin'], df_grouped[self.label])
            if (
                abs(r) == 1 and
                # check if woe for bins are monotonic
                df_grouped[self.feature].min() > self.bin_min_size
                # check if bin size is greater than 5%
                and not (df_grouped[self.feature] == df_grouped[self.label]).any()
                # check if number of good and bad is not equal to 0
            ):
                break

        return df

    @property
    def df_lite(self) -> pd.DataFrame:
        df_lite = self.__generate_correct_bins__()
        df_lite['bin'].fillna(self.missing, inplace=True)
        return df_lite[['bin', self.label]]
